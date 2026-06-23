"""
shared/constants.py

Platform-wide enumerations and numeric constants.

All severity scores, state machines, and source identifiers live here.
Never hardcode strings like "HIGH" or "CRITICAL" in agent code — import from here.

Architecture role:
- Referenced by agents, schemas, merger, DB models, and GitHub client
- Single place to change severity weights without touching agent logic
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


# ------------------------------------------------------------------ #
# Severity
# ------------------------------------------------------------------ #


class Severity(StrEnum):
    """Human-readable severity labels used throughout the platform."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class SeverityScore(IntEnum):
    """
    Numeric weights for severity.

    Used by:
    - Merger for ranking and deduplication (keep highest score)
    - Markdown report generation (sort order)
    - Auto-fix agent (threshold gate)
    """

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1

    @classmethod
    def from_severity(cls, severity: Severity | str) -> "SeverityScore":
        return cls[str(severity).upper()]


# ------------------------------------------------------------------ #
# Confidence
# ------------------------------------------------------------------ #


class ConfidenceLevel(StrEnum):
    """Confidence in a finding's accuracy, expressed as a label."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


CONFIDENCE_SCORE_MAP: dict[str, float] = {
    ConfidenceLevel.HIGH: 0.9,
    ConfidenceLevel.MEDIUM: 0.6,
    ConfidenceLevel.LOW: 0.3,
}


# ------------------------------------------------------------------ #
# Analysis Sources
# ------------------------------------------------------------------ #


class AnalysisSource(StrEnum):
    """
    Identifier for which tool produced a finding.

    Used in deduplication (same file+line+issue from multiple sources)
    and in the markdown report to attribute findings.
    """

    PYLINT = "pylint"
    FLAKE8 = "flake8"
    MYPY = "mypy"
    ESLINT = "eslint"
    BANDIT = "bandit"
    SEMGREP = "semgrep"
    LLM_REVIEW = "llm_review"
    AUTO_FIX = "auto_fix"


# ------------------------------------------------------------------ #
# Review / Job States
# ------------------------------------------------------------------ #


class ReviewStatus(StrEnum):
    """Lifecycle states for a review job."""

    PENDING = "PENDING"
    CLONING = "CLONING"
    ANALYZING = "ANALYZING"
    MERGING = "MERGING"
    POSTING = "POSTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FixStatus(StrEnum):
    """Lifecycle states for an auto-fix patch."""

    GENERATED = "GENERATED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    APPLIED = "APPLIED"
    FAILED = "FAILED"


class ApprovalAction(StrEnum):
    """Human approval actions for fix workflow."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"


# ------------------------------------------------------------------ #
# GitHub Event Types
# ------------------------------------------------------------------ #


class GitHubEventType(StrEnum):
    """GitHub webhook event action values."""

    OPENED = "opened"
    SYNCHRONIZE = "synchronize"
    REOPENED = "reopened"
    CLOSED = "closed"


# Subset of PR events that should trigger a review
REVIEW_TRIGGER_ACTIONS: frozenset[GitHubEventType] = frozenset(
    {
        GitHubEventType.OPENED,
        GitHubEventType.SYNCHRONIZE,
        GitHubEventType.REOPENED,
    }
)


# ------------------------------------------------------------------ #
# File Type Routing
# ------------------------------------------------------------------ #

# Maps file extensions to which static analyzers apply
PYTHON_EXTENSIONS: frozenset[str] = frozenset({".py", ".pyi"})
JAVASCRIPT_EXTENSIONS: frozenset[str] = frozenset({".js", ".jsx", ".mjs", ".cjs"})
TYPESCRIPT_EXTENSIONS: frozenset[str] = frozenset({".ts", ".tsx"})
JS_TS_EXTENSIONS: frozenset[str] = JAVASCRIPT_EXTENSIONS | TYPESCRIPT_EXTENSIONS


# ------------------------------------------------------------------ #
# Queue
# ------------------------------------------------------------------ #

REDIS_JOB_SCHEMA_VERSION: str = "1.0"
MAX_JOB_RETRY_COUNT: int = 3
JOB_VISIBILITY_TIMEOUT_SECONDS: int = 600  # 10 min max per review job


# ------------------------------------------------------------------ #
# Cost tracking (USD per 1M tokens, as of model release)
# ------------------------------------------------------------------ #

MODEL_COST_MAP: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 5.00, "completion": 15.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "claude-sonnet-4-6": {"prompt": 3.00, "completion": 15.00},
    "claude-haiku-4-5-20251001": {"prompt": 0.25, "completion": 1.25},
}