"""Agent loop: an explicit LangGraph state machine that retrieves schema, generates SQL, guards it, executes it, and feeds errors back."""

import re
import time
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from text_to_sql_agent.guardrail import GuardrailVerdict, check_query
from text_to_sql_agent.llm import LLM, LLMResponse
from text_to_sql_agent.sandbox import ExecutionResult, execute_query, schema_ddl
from text_to_sql_agent.schema_retrieval import SchemaRetrieval, SchemaRetriever
from text_to_sql_agent.tracing import Tracer

SYSTEM_PROMPT = """You translate natural-language questions into SQLite SQL.
Rules:
- Output exactly one SELECT statement and nothing else: no explanation, no markdown fences.
- Use only the tables and columns listed in the schema.
- Never write, alter, or attach anything; PRAGMA is forbidden.
- If feedback from a previous attempt is given, fix the specific error it describes."""

EXECUTION_HINTS = (
    ("no such column", "That column does not exist; use only column names from the schema, qualified by table where needed."),
    ("no such table", "That table does not exist; use only table names from the schema."),
    ("ambiguous column", "Qualify the column with its table name or alias."),
    ("syntax error", "Fix the SQL syntax so it is valid SQLite."),
    ("no such function", "That function is not available in SQLite; use a supported one."),
)
DEFAULT_EXECUTION_HINT = "Correct the query so it runs on SQLite."
GUARDRAIL_HINT = "Write exactly one read-only SELECT statement."

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class Observation(BaseModel):
    """Structured feedback packaged from a failed attempt and fed into the next generation prompt."""

    attempt: int
    previous_sql: str
    stage: Literal["guardrail", "execution"]
    error: str
    hint: str

    def render(self) -> str:
        """Format the observation as the feedback block shown to the model."""
        return (
            f"Attempt {self.attempt} failed at the {self.stage} stage.\n"
            f"SQL:\n{self.previous_sql}\n"
            f"Error: {self.error}\n"
            f"Hint: {self.hint}"
        )


class Attempt(BaseModel):
    """Everything recorded about one pass through generate -> guardrail -> execute."""

    index: int
    raw_response: str
    sql: str
    llm: LLMResponse
    verdict: GuardrailVerdict | None = None
    execution: ExecutionResult | None = None
    observation: Observation | None = None


class AgentState(TypedDict):
    question: str
    db_path: str
    schema_text: str
    retrieval: SchemaRetrieval | None
    max_attempts: int
    attempts: list[Attempt]
    status: Literal["running", "success", "failed"]


class AgentRun(BaseModel):
    """Complete record of one run: the input, the schema the model saw and how it was chosen, every attempt, and the outcome."""

    question: str
    db_path: str
    schema_text: str
    retrieval: SchemaRetrieval | None = None
    status: Literal["success", "failed"]
    attempts: list[Attempt]
    latency_ms: float
    trace_id: str | None = None

    @property
    def final_sql(self) -> str | None:
        return self.attempts[-1].sql if self.status == "success" else None

    @property
    def result(self) -> ExecutionResult | None:
        return self.attempts[-1].execution if self.status == "success" else None

    @property
    def prompt_tokens(self) -> int:
        return sum(attempt.llm.prompt_tokens for attempt in self.attempts)

    @property
    def completion_tokens(self) -> int:
        return sum(attempt.llm.completion_tokens for attempt in self.attempts)

    @property
    def cost_usd(self) -> float:
        return sum(attempt.llm.cost_usd for attempt in self.attempts)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply, unwrapping a markdown fence if one is present."""
    match = FENCE.search(text)
    return (match.group(1) if match else text).strip()


def build_prompt(state: AgentState) -> str:
    """Compose the schema, the question, and every prior observation into the generation prompt."""
    sections = [f"Schema:\n{state['schema_text']}", f"Question: {state['question']}"]
    feedback = [attempt.observation.render() for attempt in state["attempts"] if attempt.observation]
    if feedback:
        sections.append("Previous attempts failed. Feedback:\n\n" + "\n\n".join(feedback))
    sections.append("SQL:")
    return "\n\n".join(sections)


def package_observation(attempt: Attempt) -> Observation:
    """Turn a rejected or failed attempt into structured feedback for the next generation."""
    if attempt.execution is not None and attempt.execution.error is not None:
        return Observation(
            attempt=attempt.index,
            previous_sql=attempt.sql,
            stage="execution",
            error=attempt.execution.error,
            hint=execution_hint(attempt.execution.error),
        )
    if attempt.verdict is not None and not attempt.verdict.allowed:
        return Observation(
            attempt=attempt.index,
            previous_sql=attempt.sql,
            stage="guardrail",
            error=attempt.verdict.reason or "Query rejected by guardrail.",
            hint=GUARDRAIL_HINT,
        )
    raise ValueError(f"Attempt {attempt.index} did not fail, so there is nothing to observe.")


def execution_hint(error: str) -> str:
    """Map a raw sqlite error message to a short corrective hint."""
    lowered = error.lower()
    for needle, hint in EXECUTION_HINTS:
        if needle in lowered:
            return hint
    return DEFAULT_EXECUTION_HINT


def build_graph(llm: LLM, tracer: Tracer, retriever: SchemaRetriever | None = None) -> CompiledStateGraph:
    """Wire the retrieve -> generate -> guardrail -> execute -> observe state machine around the model, tracing every node."""

    def retrieve(state: AgentState) -> dict:
        inputs = {"question": state["question"], "db_path": state["db_path"]}
        with tracer.step("retrieve", "retriever", 0, inputs) as step:
            if retriever is None:
                step.record_output({"mode": "full_schema"})
                return {"schema_text": schema_ddl(state["db_path"]), "retrieval": None}
            retrieval = retriever.retrieve(state["db_path"], state["question"])
            step.record_retrieval(retrieval)
        return {"schema_text": retrieval.schema_text, "retrieval": retrieval}

    def generate(state: AgentState) -> dict:
        index = len(state["attempts"]) + 1
        prompt = build_prompt(state)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        with tracer.step("generate", "generation", index, messages) as step:
            response = llm.complete(SYSTEM_PROMPT, prompt)
            step.record_generation(response)
        attempt = Attempt(index=index, raw_response=response.text, sql=extract_sql(response.text), llm=response)
        return {"attempts": [*state["attempts"], attempt]}

    def guardrail(state: AgentState) -> dict:
        attempt = state["attempts"][-1]
        with tracer.step("guardrail", "guardrail", attempt.index, {"sql": attempt.sql}) as step:
            verdict = check_query(attempt.sql)
            step.record_verdict(verdict)
        return {"attempts": _update_last(state["attempts"], verdict=verdict)}

    def execute(state: AgentState) -> dict:
        attempt = state["attempts"][-1]
        with tracer.step("execute", "tool", attempt.index, {"sql": attempt.sql, "db_path": state["db_path"]}) as step:
            result = execute_query(state["db_path"], attempt.sql)
            step.record_execution(result)
        return {
            "attempts": _update_last(state["attempts"], execution=result),
            "status": "success" if result.ok else "running",
        }

    def observe(state: AgentState) -> dict:
        attempt = state["attempts"][-1]
        exhausted = len(state["attempts"]) >= state["max_attempts"]
        with tracer.step("observe", "span", attempt.index, {"max_attempts": state["max_attempts"]}) as step:
            observation = package_observation(attempt)
            step.record_output({**observation.model_dump(), "exhausted": exhausted})
        return {
            "attempts": _update_last(state["attempts"], observation=observation),
            "status": "failed" if exhausted else "running",
        }

    def route_after_guardrail(state: AgentState) -> str:
        verdict = state["attempts"][-1].verdict
        return "execute" if verdict is not None and verdict.allowed else "observe"

    def route_after_execute(state: AgentState) -> str:
        return END if state["status"] == "success" else "observe"

    def route_after_observe(state: AgentState) -> str:
        return END if state["status"] == "failed" else "generate"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("guardrail", guardrail)
    graph.add_node("execute", execute)
    graph.add_node("observe", observe)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, ["execute", "observe"])
    graph.add_conditional_edges("execute", route_after_execute, [END, "observe"])
    graph.add_conditional_edges("observe", route_after_observe, [END, "generate"])
    return graph.compile()


def run_agent(
    question: str,
    db_path: str | Path,
    llm: LLM,
    retriever: SchemaRetriever | None = None,
    max_attempts: int = 3,
    tracer: Tracer | None = None,
) -> AgentRun:
    """Run the loop for one question inside one trace; without a retriever the prompt carries the full schema DDL."""
    started = time.perf_counter()
    tracer = tracer if tracer is not None else Tracer()
    state: AgentState = {
        "question": question,
        "db_path": str(db_path),
        "schema_text": "",
        "retrieval": None,
        "max_attempts": max_attempts,
        "attempts": [],
        "status": "running",
    }
    with tracer.run(question, state["db_path"], max_attempts) as trace:
        final = build_graph(llm, tracer, retriever).invoke(state, config={"recursion_limit": 4 * max_attempts + 3})
        run = AgentRun(
            question=question,
            db_path=final["db_path"],
            schema_text=final["schema_text"],
            retrieval=final["retrieval"],
            status=final["status"],
            attempts=final["attempts"],
            latency_ms=(time.perf_counter() - started) * 1000,
            trace_id=tracer.current_trace_id(),
        )
        trace.record_output({"status": run.status, "final_sql": run.final_sql, "attempts": len(run.attempts)})
    return run


def _update_last(attempts: list[Attempt], **fields: object) -> list[Attempt]:
    """Return attempts with the most recent one replaced by a copy carrying the given fields."""
    return [*attempts[:-1], attempts[-1].model_copy(update=fields)]
