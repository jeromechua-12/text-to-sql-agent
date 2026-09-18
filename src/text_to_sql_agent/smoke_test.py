"""Smoke test: run the agent over a small Spider dev subset and report execution accuracy, recovery rate, and latency."""

import argparse
import json
import logging
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from text_to_sql_agent.agent import run_agent
from text_to_sql_agent.llm import LLM, OpenAIChat
from text_to_sql_agent.run_log import RunLogger
from text_to_sql_agent.sandbox import execute_query
from text_to_sql_agent.tracing import Tracer

DEFAULT_DEV_JSON = Path("data/spider_data/dev.json")
DEFAULT_DB_ROOT = Path("data/spider_data/database")
DEFAULT_OUTPUT_DIR = Path("logs/smoke")

logger = logging.getLogger("text_to_sql_agent.smoke_test")


class SpiderExample(BaseModel):
    """One Spider dev question with its gold SQL and the database it targets."""

    index: int
    db_id: str
    question: str
    gold_sql: str


class ExampleResult(BaseModel):
    """Outcome of one example: whether the final SQL matched the gold result set, plus the run's retry and cost summary."""

    index: int
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str | None
    status: Literal["success", "failed"]
    correct: bool
    attempt_count: int
    first_attempt_failed: bool
    recovered: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    run_id: str
    trace_id: str | None = None


class SmokeSummary(BaseModel):
    """Headline numbers over one smoke test batch."""

    examples: int
    correct: int
    execution_accuracy: float
    first_attempt_failures: int
    recovered: int
    recovery_rate: float | None
    latency_median_ms: float
    latency_p95_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class SmokeReport(BaseModel):
    """Everything one smoke test batch produced: its settings, the per-example results, and the summary."""

    batch_id: str
    recorded_at: datetime
    model: str
    max_attempts: int
    summary: SmokeSummary
    results: list[ExampleResult]


def load_examples(dev_json: str | Path, limit: int) -> list[SpiderExample]:
    """Load the Spider dev set and keep every k-th question so the subset spans the whole file and all its databases."""
    entries = json.loads(Path(dev_json).read_text(encoding="utf-8"))
    examples = [
        SpiderExample(index=index, db_id=entry["db_id"], question=entry["question"], gold_sql=entry["query"])
        for index, entry in enumerate(entries)
    ]
    if limit >= len(examples):
        return examples
    step = len(examples) // limit
    return [examples[index] for index in range(0, step * limit, step)]


def db_path_for(db_root: str | Path, db_id: str) -> Path:
    """Locate the SQLite file for a Spider database id under the Spider database directory."""
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def execution_match(db_path: str | Path, predicted_sql: str, gold_sql: str) -> bool:
    """Run both queries in the read-only sandbox and compare result sets, honouring row order only when the gold orders."""
    gold = execute_query(db_path, gold_sql, row_limit=None)
    if not gold.ok:
        raise ValueError(f"Gold SQL failed on {db_path}: {gold.error}")
    predicted = execute_query(db_path, predicted_sql, row_limit=None)
    if not predicted.ok:
        return False
    if "order by" in gold_sql.lower():
        return gold.rows == predicted.rows
    return Counter(gold.rows) == Counter(predicted.rows)


def run_smoke_test(
    examples: list[SpiderExample],
    db_root: str | Path,
    llm: LLM,
    run_logger: RunLogger,
    tracer: Tracer | None = None,
    max_attempts: int = 3,
    batch_id: str | None = None,
) -> list[ExampleResult]:
    """Run the agent on each example, score it against the gold result set, and log every run with its score attached."""
    tracer = tracer if tracer is not None else Tracer()
    results = []
    for position, example in enumerate(examples, start=1):
        db_path = db_path_for(db_root, example.db_id)
        run = run_agent(example.question, db_path, llm, max_attempts=max_attempts, tracer=tracer)
        correct = run.final_sql is not None and execution_match(db_path, run.final_sql, example.gold_sql)
        record = run_logger.log(
            run,
            batch_id=batch_id,
            spider_index=example.index,
            db_id=example.db_id,
            gold_sql=example.gold_sql,
            correct=correct,
        )
        result = ExampleResult(
            index=example.index,
            db_id=example.db_id,
            question=example.question,
            gold_sql=example.gold_sql,
            predicted_sql=run.final_sql,
            status=run.status,
            correct=correct,
            attempt_count=record.attempt_count,
            first_attempt_failed=run.attempts[0].observation is not None,
            recovered=record.recovered,
            latency_ms=run.latency_ms,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            cost_usd=record.cost_usd,
            run_id=record.run_id,
            trace_id=record.trace_id,
        )
        results.append(result)
        logger.info(
            "example %d/%d spider_index=%d db=%s status=%s correct=%s attempts=%d latency_ms=%.0f",
            position,
            len(examples),
            example.index,
            example.db_id,
            result.status,
            result.correct,
            result.attempt_count,
            result.latency_ms,
        )
    return results


def summarise(results: list[ExampleResult]) -> SmokeSummary:
    """Aggregate per-example results into execution accuracy, recovery rate, latency percentiles, and usage totals."""
    correct = sum(result.correct for result in results)
    first_attempt_failures = sum(result.first_attempt_failed for result in results)
    recovered = sum(result.recovered for result in results)
    latencies = sorted(result.latency_ms for result in results)
    return SmokeSummary(
        examples=len(results),
        correct=correct,
        execution_accuracy=correct / len(results),
        first_attempt_failures=first_attempt_failures,
        recovered=recovered,
        recovery_rate=recovered / first_attempt_failures if first_attempt_failures else None,
        latency_median_ms=statistics.median(latencies),
        latency_p95_ms=percentile(latencies, 0.95),
        prompt_tokens=sum(result.prompt_tokens for result in results),
        completion_tokens=sum(result.completion_tokens for result in results),
        cost_usd=sum(result.cost_usd for result in results),
    )


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an ascending list of values."""
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def format_summary(report: SmokeReport) -> str:
    """Render the report's headline numbers as a few human-readable lines."""
    summary = report.summary
    recovery = (
        f"{summary.recovered}/{summary.first_attempt_failures} first-attempt failures rescued = {summary.recovery_rate:.1%}"
        if summary.recovery_rate is not None
        else "no first-attempt failures"
    )
    return "\n".join(
        [
            f"Smoke test {report.batch_id}: {summary.examples} Spider dev questions, model {report.model}, max {report.max_attempts} attempts",
            f"Execution accuracy: {summary.correct}/{summary.examples} = {summary.execution_accuracy:.1%}",
            f"Self-correction recovery: {recovery}",
            f"Latency: median {summary.latency_median_ms:.0f} ms, p95 {summary.latency_p95_ms:.0f} ms",
            f"Tokens: {summary.prompt_tokens} prompt, {summary.completion_tokens} completion; cost ${summary.cost_usd:.4f}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: sample the dev set, run the batch, write the run log and report, and print the summary."""
    parser = argparse.ArgumentParser(description="Run the text-to-SQL agent over a small Spider dev subset.")
    parser.add_argument("--n", type=int, default=40, help="number of dev questions to run")
    parser.add_argument("--dev-json", type=Path, default=DEFAULT_DEV_JSON)
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = args.output_dir / batch_id
    examples = load_examples(args.dev_json, args.n)
    results = run_smoke_test(
        examples,
        args.db_root,
        OpenAIChat(args.model),
        RunLogger(batch_dir / "runs.jsonl"),
        tracer=Tracer(),
        max_attempts=args.max_attempts,
        batch_id=batch_id,
    )
    report = SmokeReport(
        batch_id=batch_id,
        recorded_at=datetime.now(UTC),
        model=args.model,
        max_attempts=args.max_attempts,
        summary=summarise(results),
        results=results,
    )
    report_path = batch_dir / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(format_summary(report))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
