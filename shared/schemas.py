"""
shared/schemas.py

Canonical Pydantic v2 schemas for the AI Code Review Platform.

These schemas ARE the API contract between all components:
  - Webhook API validates inbound GitHub payloads into these types
  - Queue serializes/deserializes jobs using these types
  - Every agent accepts and returns these types
  - Storage layer maps these types to SQLAlchemy models
  - GitHub client reads these types to post comments

Design principles:
  - All fields explicitly typed — no Dict[str, Any] in hot paths
  - model_config = ConfigDict(frozen=True) on immutable transfer objects
  - Mutable aggregates (ReviewResult) use plain ConfigDict
  - Timestamps always UTC
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from shared.constants import (
    AnalysisSource,
    ApprovalAction,
    ConfidenceLevel,
    FixStatus,
    ReviewStatus,
    Severity,
)


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


# ------------------------------------------------------------------ #
# GitHub Webhook Payload
# ------------------------------------------------------------------ #


class GitHubUserSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    login: str
    id: int
    html_url: str


class GitHubRepoSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    full_name: str
    clone_url: str
    default_branch: str = "main"
    language: str | None = None


class GitHubPRSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    number: int
    title: str
    body: str | None = None
    state: str
    html_url: str
    head_sha: str = Field(alias="head_sha_value")
    base_sha: str = Field(alias="base_sha_value")
    head_branch: str
    base_branch: str
    author: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(frozen=True, populate_by_name=True)


class WebhookPayloadSchema(BaseModel):
    """
    Validated GitHub pull_request webhook payload.

    Parsed by the webhook route handler before pushing to Redis.
    Contains only the fields needed downstream — not the raw GitHub blob.
    """

    model_config = ConfigDict(frozen=True)

    action: str
    installation_id: int | None = None

    # Flattened from nested GitHub payload
    repo_id: int
    repo_full_name: str
    repo_clone_url: str
    repo_default_branch: str
    repo_language: str | None = None

    pr_number: int
    pr_title: str
    pr_body: str | None = None
    pr_html_url: str
    pr_head_sha: str
    pr_base_sha: str
    pr_head_branch: str
    pr_base_branch: str
    pr_author: str

    delivery_id: str | None = None
    received_at: datetime = Field(default_factory=utcnow)


# ------------------------------------------------------------------ #
# Queue Job
# ------------------------------------------------------------------ #


class ReviewJobSchema(BaseModel):
    """
    Job envelope pushed to Redis and consumed by the worker.

    The worker reconstructs context entirely from this object —
    no shared in-memory state between webhook API and worker.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str = Field(default_factory=new_uuid)
    schema_version: str = "1.0"
    created_at: datetime = Field(default_factory=utcnow)
    retry_count: int = 0

    payload: WebhookPayloadSchema


# ------------------------------------------------------------------ #
# Diff / File
# ------------------------------------------------------------------ #


class FileChangeSchema(BaseModel):
    """A single changed file extracted from the PR diff."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    change_type: str  # "added" | "modified" | "deleted" | "renamed"
    patch: str | None = None  # Unified diff patch text
    additions: int = 0
    deletions: int = 0
    language: str | None = None  # Inferred from extension


class PRDiffSchema(BaseModel):
    """Complete diff for a PR — input to all analysis agents."""

    model_config = ConfigDict(frozen=True)

    repo_full_name: str
    pr_number: int
    head_sha: str
    base_sha: str
    local_repo_path: str  # Absolute path to cloned repo on disk
    changed_files: list[FileChangeSchema]

    @property
    def changed_file_paths(self) -> list[str]:
        return [f.file_path for f in self.changed_files]


# ------------------------------------------------------------------ #
# Finding (core output of every analysis agent)
# ------------------------------------------------------------------ #


class FindingSchema(BaseModel):
    """
    Normalized finding produced by any analysis agent.

    This is the central schema of the entire platform.

    All agents — static, security, LLM — MUST return findings
    in this exact shape. The merger operates on lists of FindingSchema.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_uuid)

    # Attribution
    source: AnalysisSource
    rule_id: str | None = None  # Tool-specific rule identifier

    # Location
    file_path: str
    line_number: int | None = None
    end_line_number: int | None = None
    column: int | None = None

    # Classification
    severity: Severity
    confidence: ConfidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)

    # Content
    issue: str = Field(description="Short description of the problem")
    suggestion: str = Field(description="Actionable fix recommendation")
    code_snippet: str | None = None

    # Optional SARIF reference
    sarif_rule_url: str | None = None

    # Deduplication key (set by merger)
    dedup_key: str | None = None

    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("confidence_score")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


# ------------------------------------------------------------------ #
# Agent Results
# ------------------------------------------------------------------ #


class StaticAnalysisResult(BaseModel):
    """Output of the Static Analysis Agent."""

    findings: list[FindingSchema] = Field(default_factory=list)
    files_analyzed: list[str] = Field(default_factory=list)
    analyzers_run: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


class SecurityAnalysisResult(BaseModel):
    """Output of the Security Analysis Agent."""

    findings: list[FindingSchema] = Field(default_factory=list)
    files_analyzed: list[str] = Field(default_factory=list)
    analyzers_run: list[str] = Field(default_factory=list)
    sarif_reports: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


class LLMReviewResult(BaseModel):
    """Output of the LLM Review Agent."""

    findings: list[FindingSchema] = Field(default_factory=list)
    files_reviewed: list[str] = Field(default_factory=list)
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


class MergedReviewResult(BaseModel):
    """
    Output of the Merger Agent — final consolidated review.

    Input to GitHub client for posting comments and to storage for persistence.
    """

    findings: list[FindingSchema] = Field(default_factory=list)
    deduplicated_count: int = 0
    total_before_dedup: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    markdown_report: str = ""
    duration_seconds: float = 0.0


# ------------------------------------------------------------------ #
# Auto-Fix
# ------------------------------------------------------------------ #


class PatchSchema(BaseModel):
    """A generated code patch for a specific finding."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_uuid)
    finding_id: str
    file_path: str

    # Unified diff format patch
    unified_diff: str

    # Human-readable explanation
    explanation: str

    # Validation results
    tests_passed: bool | None = None
    lint_passed: bool | None = None
    validation_errors: list[str] = Field(default_factory=list)

    status: FixStatus = FixStatus.GENERATED
    created_at: datetime = Field(default_factory=utcnow)


class FixRequestSchema(BaseModel):
    """Request to the Auto-Fix Agent."""

    model_config = ConfigDict(frozen=True)

    finding: FindingSchema
    repo_path: str
    repo_full_name: str
    pr_number: int
    base_branch: str


class FixApprovalSchema(BaseModel):
    """Human approval/rejection of a generated fix."""

    patch_id: str
    action: ApprovalAction
    reviewer: str
    comment: str | None = None
    reviewed_at: datetime = Field(default_factory=utcnow)


# ------------------------------------------------------------------ #
# LLM Usage
# ------------------------------------------------------------------ #


class LLMUsageSchema(BaseModel):
    """Token usage and cost for a single LLM call. Persisted to PostgreSQL."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=new_uuid)
    review_id: str
    agent_name: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    called_at: datetime = Field(default_factory=utcnow)


# ------------------------------------------------------------------ #
# Pipeline State (LangGraph state object)
# ------------------------------------------------------------------ #


class PipelineState(BaseModel):
    """
    LangGraph state object passed between graph nodes.

    Mutable — each node enriches or replaces fields.
    LangGraph requires state to be a TypedDict or Pydantic model.

    NOT frozen — nodes update it in place via state merging.
    """

    # Input
    job: ReviewJobSchema
    diff: PRDiffSchema | None = None

    # Agent outputs (populated concurrently)
    static_result: StaticAnalysisResult | None = None
    security_result: SecurityAnalysisResult | None = None
    llm_result: LLMReviewResult | None = None

    # Merged output
    merged_result: MergedReviewResult | None = None

    # Fix patches (populated after merger)
    patches: list[PatchSchema] = Field(default_factory=list)

    # Job metadata
    review_id: str = Field(default_factory=new_uuid)
    status: ReviewStatus = ReviewStatus.PENDING
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None


# ------------------------------------------------------------------ #
# API Response Schemas
# ------------------------------------------------------------------ #


class WebhookAckResponse(BaseModel):
    """Immediate acknowledgement returned by webhook endpoint."""

    job_id: str
    message: str = "Review job queued"
    queued_at: datetime = Field(default_factory=utcnow)


class ReviewSummaryResponse(BaseModel):
    """Public summary of a completed review (for API consumers)."""

    review_id: str
    pr_number: int
    repo_full_name: str
    status: ReviewStatus
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    info_count: int
    markdown_report: str
    created_at: datetime