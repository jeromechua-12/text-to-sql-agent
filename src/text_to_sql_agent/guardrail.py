"""Static SQL guardrail: parse with sqlglot and allow only a single read-only SELECT.
Sits between generation and execution and returns a structured allow/reject verdict."""

from enum import Enum

import sqlglot
from pydantic import BaseModel
from sqlglot import errors, exp

READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)

FORBIDDEN_ROOTS = {
    exp.Insert: "write statement (INSERT)",
    exp.Update: "write statement (UPDATE)",
    exp.Delete: "write statement (DELETE)",
    exp.Merge: "write statement (MERGE)",
    exp.Create: "DDL statement (CREATE)",
    exp.Drop: "DDL statement (DROP)",
    exp.Alter: "DDL statement (ALTER)",
    exp.TruncateTable: "DDL statement (TRUNCATE)",
    exp.Pragma: "PRAGMA statement",
    exp.Attach: "ATTACH statement",
    exp.Detach: "DETACH statement",
}


class GuardrailDecision(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"


class RejectionReason(str, Enum):
    EMPTY_QUERY = "empty_query"
    PARSE_ERROR = "parse_error"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_A_SELECT = "not_a_select"
    FORBIDDEN_STATEMENT = "forbidden_statement"


class GuardrailVerdict(BaseModel):
    """Structured outcome of a guardrail check, logged and fed back into the correction loop."""

    decision: GuardrailDecision
    sql: str
    reason_code: RejectionReason | None = None
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision is GuardrailDecision.ALLOW


def _reject(sql: str, code: RejectionReason, reason: str) -> GuardrailVerdict:
    return GuardrailVerdict(decision=GuardrailDecision.REJECT, sql=sql, reason_code=code, reason=reason)


def check_query(sql: str, dialect: str = "sqlite") -> GuardrailVerdict:
    """Parse sql and allow it only if it is exactly one read-only SELECT statement."""
    if not sql or not sql.strip():
        return _reject(sql, RejectionReason.EMPTY_QUERY, "Query is empty.")

    try:
        statements = [stmt for stmt in sqlglot.parse(sql, dialect=dialect) if stmt is not None]
    except errors.ParseError as err:
        return _reject(sql, RejectionReason.PARSE_ERROR, f"Query failed to parse: {err}")

    if not statements:
        return _reject(sql, RejectionReason.EMPTY_QUERY, "Query is empty.")

    if len(statements) > 1:
        return _reject(
            sql,
            RejectionReason.MULTIPLE_STATEMENTS,
            f"Only a single statement is allowed, found {len(statements)}.",
        )

    root = statements[0]
    forbidden = _find_forbidden(root)
    if forbidden is not None:
        return _reject(sql, RejectionReason.FORBIDDEN_STATEMENT, f"Rejected {forbidden}; only read-only SELECT is allowed.")

    if not isinstance(root, READ_ONLY_ROOTS):
        return _reject(
            sql,
            RejectionReason.NOT_A_SELECT,
            f"Query must be a SELECT, found {type(root).__name__.upper()}.",
        )

    return GuardrailVerdict(decision=GuardrailDecision.ALLOW, sql=sql)


def _find_forbidden(root: exp.Expr) -> str | None:
    """Return a description of the first forbidden node found anywhere in the tree, else None."""
    for node in root.walk():
        description = FORBIDDEN_ROOTS.get(type(node))
        if description is not None:
            return description
    return None
