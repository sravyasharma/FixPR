"""
storage/repositories.py

Repository Pattern implementation for the AI Code Review Platform.

Each repository class:
  - Accepts an AsyncSession in __init__ (injected, not created internally)
  - Exposes typed async methods (no raw SQL in agents or routes)
  - Translates between Pydantic schemas and ORM models
  - Handles upserts where needed (re-reviewed PRs, known repos)

Architecture role:
  - Agents and API never import ORM models directly
  - Worker creates repositories once per job using a single session
  - Test fixtures can inject a mock session or test DB session
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.constants import FixStatus, ReviewStatus
from shared.logger import get_logger
from shared.schemas import (
    FindingSchema,
    FixApprovalSchema,
    LLMUsageSchema,
    MergedReviewResult,
    PatchSchema,
    WebhookPayloadSchema,
)
from storage.models import (
    FindingModel,
    FixPatchModel,
    LLMUsageModel,
    PullRequestModel,
    RepositoryModel,
    ReviewModel,
)

logger = get_logger(__name__)


class RepositoryRepo:
    """CRUD for GitHub repositories."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, payload: WebhookPayloadSchema) -> RepositoryModel:
        """
        Upsert a repository record.

        GitHub repo IDs are stable — use them as the idempotency key.
        """
        stmt = select(RepositoryModel).where(
            RepositoryModel.github_repo_id == payload.repo_id
        )
        result = await self._session.execute(stmt)
        repo = result.scalar_one_or_none()

        if repo is None:
            repo = RepositoryModel(
                github_repo_id=payload.repo_id,
                full_name=payload.repo_full_name,
                clone_url=payload.repo_clone_url,
                default_branch=payload.repo_default_branch,
                language=payload.repo_language,
            )
            self._session.add(repo)
            await self._session.flush()
            logger.info("Repository created", full_name=payload.repo_full_name)
        else:
            # Update mutable fields in case they changed
            repo.clone_url = payload.repo_clone_url
            repo.language = payload.repo_language
            logger.debug("Repository found", full_name=payload.repo_full_name)

        return repo

    async def get_by_full_name(self, full_name: str) -> RepositoryModel | None:
        stmt = select(RepositoryModel).where(RepositoryModel.full_name == full_name)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


class PullRequestRepo:
    """CRUD for pull requests."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(
        self, payload: WebhookPayloadSchema, repository_id: uuid.UUID
    ) -> PullRequestModel:
        """Upsert a PR record. Re-reviews update head_sha."""
        stmt = select(PullRequestModel).where(
            PullRequestModel.repository_id == repository_id,
            PullRequestModel.pr_number == payload.pr_number,
        )
        result = await self._session.execute(stmt)
        pr = result.scalar_one_or_none()

        if pr is None:
            pr = PullRequestModel(
                repository_id=repository_id,
                pr_number=payload.pr_number,
                title=payload.pr_title,
                body=payload.pr_body,
                author=payload.pr_author,
                head_sha=payload.pr_head_sha,
                base_sha=payload.pr_base_sha,
                head_branch=payload.pr_head_branch,
                base_branch=payload.pr_base_branch,
                html_url=payload.pr_html_url,
            )
            self._session.add(pr)
            await self._session.flush()
            logger.info("PullRequest created", pr_number=payload.pr_number)
        else:
            # New commit pushed — update SHA and title
            pr.head_sha = payload.pr_head_sha
            pr.title = payload.pr_title
            pr.body = payload.pr_body
            logger.debug("PullRequest updated", pr_number=payload.pr_number)

        return pr

    async def get_by_id(self, pr_id: uuid.UUID) -> PullRequestModel | None:
        return await self._session.get(PullRequestModel, pr_id)


class ReviewRepo:
    """CRUD for review runs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, pr_id: uuid.UUID, job_id: str, review_id: str
    ) -> ReviewModel:
        """Create a new review record when a job starts."""
        review = ReviewModel(
            id=uuid.UUID(review_id),
            pull_request_id=pr_id,
            job_id=job_id,
            status=ReviewStatus.PENDING,
            started_at=datetime.now(tz=timezone.utc),
        )
        self._session.add(review)
        await self._session.flush()
        logger.info("Review created", review_id=review_id, job_id=job_id)
        return review

    async def update_status(self, review_id: str, status: ReviewStatus) -> None:
        stmt = (
            update(ReviewModel)
            .where(ReviewModel.id == uuid.UUID(review_id))
            .values(status=status)
        )
        await self._session.execute(stmt)

    async def complete(
        self, review_id: str, result: MergedReviewResult
    ) -> None:
        """Mark review as completed and store summary counts."""
        stmt = (
            update(ReviewModel)
            .where(ReviewModel.id == uuid.UUID(review_id))
            .values(
                status=ReviewStatus.COMPLETED,
                completed_at=datetime.now(tz=timezone.utc),
                duration_seconds=result.duration_seconds,
                total_findings=len(result.findings),
                critical_count=result.critical_count,
                high_count=result.high_count,
                medium_count=result.medium_count,
                low_count=result.low_count,
                info_count=result.info_count,
                markdown_report=result.markdown_report,
            )
        )
        await self._session.execute(stmt)

    async def fail(self, review_id: str, error_message: str) -> None:
        stmt = (
            update(ReviewModel)
            .where(ReviewModel.id == uuid.UUID(review_id))
            .values(
                status=ReviewStatus.FAILED,
                completed_at=datetime.now(tz=timezone.utc),
                error_message=error_message[:2048],  # truncate
            )
        )
        await self._session.execute(stmt)

    async def get_by_id(self, review_id: str) -> ReviewModel | None:
        return await self._session.get(ReviewModel, uuid.UUID(review_id))

    async def get_by_job_id(self, job_id: str) -> ReviewModel | None:
        stmt = select(ReviewModel).where(ReviewModel.job_id == job_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


class FindingRepo:
    """Bulk insert and query for findings."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_insert(
        self, review_id: str, findings: list[FindingSchema]
    ) -> list[FindingModel]:
        """
        Insert all findings for a review in one flush.

        Uses the finding's own UUID so idempotent re-inserts can be
        detected by the dedup_key constraint if needed.
        """
        models: list[FindingModel] = []
        for f in findings:
            model = FindingModel(
                id=uuid.UUID(f.id),
                review_id=uuid.UUID(review_id),
                source=f.source,
                rule_id=f.rule_id,
                file_path=f.file_path,
                line_number=f.line_number,
                end_line_number=f.end_line_number,
                column=f.column,
                severity=f.severity,
                confidence=f.confidence,
                confidence_score=f.confidence_score,
                issue=f.issue,
                suggestion=f.suggestion,
                code_snippet=f.code_snippet,
                dedup_key=f.dedup_key,
                sarif_rule_url=f.sarif_rule_url,
            )
            models.append(model)
            self._session.add(model)

        await self._session.flush()
        logger.info("Findings inserted", count=len(models), review_id=review_id)
        return models

    async def get_by_review(self, review_id: str) -> list[FindingModel]:
        stmt = (
            select(FindingModel)
            .where(FindingModel.review_id == uuid.UUID(review_id))
            .order_by(FindingModel.severity.desc(), FindingModel.confidence_score.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class LLMUsageRepo:
    """Insert and aggregate LLM usage records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(self, usage: LLMUsageSchema) -> LLMUsageModel:
        model = LLMUsageModel(
            id=uuid.UUID(usage.id),
            review_id=uuid.UUID(usage.review_id),
            agent_name=usage.agent_name,
            model_name=usage.model_name,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
        )
        self._session.add(model)
        await self._session.flush()
        return model

    async def get_by_review(self, review_id: str) -> list[LLMUsageModel]:
        stmt = select(LLMUsageModel).where(
            LLMUsageModel.review_id == uuid.UUID(review_id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class FixPatchRepo:
    """CRUD for auto-fix patches and approval workflow."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, review_id: str, finding_id: str, patch: PatchSchema
    ) -> FixPatchModel:
        model = FixPatchModel(
            id=uuid.UUID(patch.id),
            review_id=uuid.UUID(review_id),
            finding_id=uuid.UUID(finding_id),
            file_path=patch.file_path,
            unified_diff=patch.unified_diff,
            explanation=patch.explanation,
            tests_passed=patch.tests_passed,
            lint_passed=patch.lint_passed,
            validation_errors=json.dumps(patch.validation_errors),
            status=FixStatus.PENDING_APPROVAL,
        )
        self._session.add(model)
        await self._session.flush()
        logger.info("FixPatch created", patch_id=patch.id, finding_id=finding_id)
        return model

    async def apply_approval(self, approval: FixApprovalSchema) -> FixPatchModel | None:
        model = await self._session.get(FixPatchModel, uuid.UUID(approval.patch_id))
        if model is None:
            logger.warning("FixPatch not found for approval", patch_id=approval.patch_id)
            return None

        model.status = approval.action  # APPROVED or REJECTED
        model.reviewer = approval.reviewer
        model.reviewer_comment = approval.comment
        model.reviewed_at = approval.reviewed_at
        await self._session.flush()
        logger.info(
            "FixPatch approval recorded",
            patch_id=approval.patch_id,
            action=approval.action,
        )
        return model

    async def mark_applied(
        self, patch_id: str, fix_branch: str, fix_pr_url: str, fix_pr_number: int
    ) -> None:
        stmt = (
            update(FixPatchModel)
            .where(FixPatchModel.id == uuid.UUID(patch_id))
            .values(
                status=FixStatus.APPLIED,
                fix_branch=fix_branch,
                fix_pr_url=fix_pr_url,
                fix_pr_number=fix_pr_number,
            )
        )
        await self._session.execute(stmt)

    async def get_approved_patches(self, review_id: str) -> list[FixPatchModel]:
        stmt = select(FixPatchModel).where(
            FixPatchModel.review_id == uuid.UUID(review_id),
            FixPatchModel.status == FixStatus.APPROVED,
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())