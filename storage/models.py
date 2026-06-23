"""
storage/models.py

SQLAlchemy 2.x async ORM models for the AI Code Review Platform.

Models:
  1. Repository     — GitHub repo metadata
  2. PullRequest    — PR being reviewed
  3. Review         — One review run per PR event
  4. Finding        — Individual issue found during review
  5. LLMUsage       — Token usage and cost per LLM call
  6. FixPatch       — Generated auto-fix patch per finding

All models use:
  - UUID primary keys (distributed-safe, no sequence contention)
  - Explicit __tablename__ (no magic naming)
  - server_default for timestamps (DB-side, not Python-side)
  - Proper FK relationships with back_populates
  - Indexes on high-cardinality query columns

Architecture role:
  - Owned exclusively by the storage/ layer
  - Agents and API never import models directly — use repositories
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ------------------------------------------------------------------ #
# Repository
# ------------------------------------------------------------------ #


class RepositoryModel(Base):
    """
    GitHub repository tracked by the platform.

    One row per GitHub repo. Multiple PRs can belong to one repo.
    """

    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    github_repo_id: Mapped[int] = mapped_column(
        Integer, unique=True, nullable=False, index=True,
        comment="GitHub's numeric repo ID"
    )
    full_name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True,
        comment="org/repo format"
    )
    clone_url: Mapped[str] = mapped_column(String(512), nullable=False)
    default_branch: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default="main"
    )
    language: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    pull_requests: Mapped[list[PullRequestModel]] = relationship(
        "PullRequestModel", back_populates="repository", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Repository {self.full_name}>"


# ------------------------------------------------------------------ #
# PullRequest
# ------------------------------------------------------------------ #


class PullRequestModel(Base):
    """
    A specific PR under review.

    A PR can have multiple reviews if it is updated (synchronize event)
    while a previous review is in progress.
    """

    __tablename__ = "pull_requests"
    __table_args__ = (
        UniqueConstraint("repository_id", "pr_number", name="uq_pr_repo_number"),
        Index("ix_pull_requests_repo_pr", "repository_id", "pr_number"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    pr_number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    head_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    base_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    head_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    base_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    html_url: Mapped[str] = mapped_column(String(512), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    repository: Mapped[RepositoryModel] = relationship(
        "RepositoryModel", back_populates="pull_requests"
    )
    reviews: Mapped[list[ReviewModel]] = relationship(
        "ReviewModel", back_populates="pull_request", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PullRequest #{self.pr_number}>"


# ------------------------------------------------------------------ #
# Review
# ------------------------------------------------------------------ #


class ReviewModel(Base):
    """
    One complete review execution.

    Created when a job is dequeued and updated as the pipeline progresses.
    Linked to all findings discovered in this run.
    """

    __tablename__ = "reviews"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pull_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("pull_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, index=True,
        comment="Redis job ID"
    )

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="PENDING", index=True
    )

    # Agent timing
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Summary counts (denormalized for fast API responses)
    total_findings: Mapped[int] = mapped_column(Integer, default=0)
    critical_count: Mapped[int] = mapped_column(Integer, default=0)
    high_count: Mapped[int] = mapped_column(Integer, default=0)
    medium_count: Mapped[int] = mapped_column(Integer, default=0)
    low_count: Mapped[int] = mapped_column(Integer, default=0)
    info_count: Mapped[int] = mapped_column(Integer, default=0)

    # Generated report
    markdown_report: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Error info if failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    pull_request: Mapped[PullRequestModel] = relationship(
        "PullRequestModel", back_populates="reviews"
    )
    findings: Mapped[list[FindingModel]] = relationship(
        "FindingModel", back_populates="review", cascade="all, delete-orphan"
    )
    llm_usages: Mapped[list[LLMUsageModel]] = relationship(
        "LLMUsageModel", back_populates="review", cascade="all, delete-orphan"
    )
    fix_patches: Mapped[list[FixPatchModel]] = relationship(
        "FixPatchModel", back_populates="review", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Review {self.id} status={self.status}>"


# ------------------------------------------------------------------ #
# Finding
# ------------------------------------------------------------------ #


class FindingModel(Base):
    """
    A single normalized finding from any analysis agent.

    The dedup_key prevents re-inserting the same logical finding
    if a PR is re-reviewed after a minor update.
    """

    __tablename__ = "findings"
    __table_args__ = (
        Index("ix_findings_review_severity", "review_id", "severity"),
        Index("ix_findings_file_line", "file_path", "line_number"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Attribution
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rule_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Location
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    line_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_line_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Classification
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Content
    issue: Mapped[str] = mapped_column(Text, nullable=False)
    suggestion: Mapped[str] = mapped_column(Text, nullable=False)
    code_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Deduplication
    dedup_key: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)

    # SARIF
    sarif_rule_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    review: Mapped[ReviewModel] = relationship("ReviewModel", back_populates="findings")
    fix_patches: Mapped[list[FixPatchModel]] = relationship(
        "FixPatchModel", back_populates="finding"
    )

    def __repr__(self) -> str:
        return f"<Finding {self.source}:{self.file_path}:{self.line_number} [{self.severity}]>"


# ------------------------------------------------------------------ #
# LLMUsage
# ------------------------------------------------------------------ #


class LLMUsageModel(Base):
    """
    Token usage and cost tracking per LLM API call.

    One row per LLM invocation. Linked to the review that triggered it.
    Aggregated by the cost tracker for billing/budget reporting.
    """

    __tablename__ = "llm_usages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)

    called_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    review: Mapped[ReviewModel] = relationship("ReviewModel", back_populates="llm_usages")

    def __repr__(self) -> str:
        return f"<LLMUsage {self.model_name} ${self.cost_usd:.4f}>"


# ------------------------------------------------------------------ #
# FixPatch
# ------------------------------------------------------------------ #


class FixPatchModel(Base):
    """
    A generated auto-fix patch for a specific finding.

    Tracks the full approval lifecycle: GENERATED → PENDING_APPROVAL → APPROVED/REJECTED → APPLIED.
    """

    __tablename__ = "fix_patches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    finding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("findings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    unified_diff: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)

    # Validation
    tests_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    lint_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    validation_errors: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="JSON array of error strings"
    )

    # Approval workflow
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="GENERATED", index=True
    )
    reviewer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reviewer_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # GitHub branch/PR created after approval
    fix_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    fix_pr_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    fix_pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    review: Mapped[ReviewModel] = relationship("ReviewModel", back_populates="fix_patches")
    finding: Mapped[FindingModel] = relationship("FindingModel", back_populates="fix_patches")

    def __repr__(self) -> str:
        return f"<FixPatch {self.id} status={self.status}>"