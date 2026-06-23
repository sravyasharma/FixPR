"""
apps/webhook_api/routes/reviews.py

API routes for querying review results and managing fix approvals.

Endpoints:
  GET  /reviews/{review_id}          — Get review summary and findings
  GET  /reviews/{review_id}/findings — Get paginated findings list
  POST /reviews/{review_id}/patches/{patch_id}/approve — Approve a fix
  POST /reviews/{review_id}/patches/{patch_id}/reject  — Reject a fix
  POST /reviews/{review_id}/patches/{patch_id}/apply   — Apply approved fix to GitHub

Architecture role:
  - Used by CI/CD systems and the approval UI to query and act on reviews
  - Fix approval triggers the GitHub fix PR workflow
"""

from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from github.branch_manager import generate_fix_branch_name, fix_commit_message
from github.github_client import GitHubClient
from shared.constants import ApprovalAction, FixStatus
from shared.logger import get_logger
from shared.schemas import FixApprovalSchema, ReviewSummaryResponse
from storage.db import get_db_session
from storage.repositories import FindingRepo, FixPatchRepo, ReviewRepo

logger = get_logger(__name__)
router = APIRouter(prefix="/reviews", tags=["reviews"])


class ApprovalRequest(BaseModel):
    reviewer: str
    comment: str | None = None


class RejectRequest(BaseModel):
    reviewer: str
    reason: str | None = None


# ------------------------------------------------------------------ #
# Review Query
# ------------------------------------------------------------------ #


@router.get("/{review_id}", response_model=ReviewSummaryResponse)
async def get_review(review_id: str) -> ReviewSummaryResponse:
    """Get the summary of a completed review."""
    async with get_db_session() as session:
        repo = ReviewRepo(session)
        review = await repo.get_by_id(review_id)

        if review is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Review {review_id} not found",
            )

        pr = review.pull_request

        return ReviewSummaryResponse(
            review_id=str(review.id),
            pr_number=pr.pr_number,
            repo_full_name=pr.repository.full_name,
            status=review.status,
            total_findings=review.total_findings,
            critical_count=review.critical_count,
            high_count=review.high_count,
            medium_count=review.medium_count,
            low_count=review.low_count,
            info_count=review.info_count,
            markdown_report=review.markdown_report or "",
            created_at=review.created_at,
        )


@router.get("/{review_id}/findings")
async def get_findings(review_id: str, severity: str | None = None) -> dict:
    """Get all findings for a review, optionally filtered by severity."""
    async with get_db_session() as session:
        repo = FindingRepo(session)
        findings = await repo.get_by_review(review_id)

        if severity:
            findings = [f for f in findings if f.severity == severity.upper()]

        return {
            "review_id": review_id,
            "total": len(findings),
            "findings": [
                {
                    "id": str(f.id),
                    "source": f.source,
                    "rule_id": f.rule_id,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "severity": f.severity,
                    "confidence": f.confidence,
                    "confidence_score": f.confidence_score,
                    "issue": f.issue,
                    "suggestion": f.suggestion,
                }
                for f in findings
            ],
        }


# ------------------------------------------------------------------ #
# Fix Approval Workflow
# ------------------------------------------------------------------ #


@router.post("/{review_id}/patches/{patch_id}/approve", status_code=status.HTTP_200_OK)
async def approve_fix(
    review_id: str, patch_id: str, body: ApprovalRequest
) -> dict:
    """
    Approve a generated fix patch.

    Changes patch status: PENDING_APPROVAL → APPROVED.
    The patch is ready to be applied to GitHub via /apply.
    """
    async with get_db_session() as session:
        repo = FixPatchRepo(session)

        approval = FixApprovalSchema(
            patch_id=patch_id,
            action=ApprovalAction.APPROVE,
            reviewer=body.reviewer,
            comment=body.comment,
        )
        patch = await repo.apply_approval(approval)

        if patch is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patch {patch_id} not found",
            )

        logger.info(
            "Fix patch approved",
            patch_id=patch_id,
            reviewer=body.reviewer,
            review_id=review_id,
        )
        return {"status": "approved", "patch_id": patch_id}


@router.post("/{review_id}/patches/{patch_id}/reject", status_code=status.HTTP_200_OK)
async def reject_fix(
    review_id: str, patch_id: str, body: RejectRequest
) -> dict:
    """
    Reject a generated fix patch.

    Changes patch status: PENDING_APPROVAL → REJECTED.
    """
    async with get_db_session() as session:
        repo = FixPatchRepo(session)

        approval = FixApprovalSchema(
            patch_id=patch_id,
            action=ApprovalAction.REJECT,
            reviewer=body.reviewer,
            comment=body.reason,
        )
        patch = await repo.apply_approval(approval)

        if patch is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patch {patch_id} not found",
            )

        logger.info(
            "Fix patch rejected",
            patch_id=patch_id,
            reviewer=body.reviewer,
            review_id=review_id,
        )
        return {"status": "rejected", "patch_id": patch_id}


@router.post("/{review_id}/patches/{patch_id}/apply", status_code=status.HTTP_200_OK)
async def apply_fix(review_id: str, patch_id: str) -> dict:
    """
    Apply an APPROVED patch to GitHub.

    Workflow:
      1. Load patch from DB — verify status=APPROVED
      2. Create fix branch on GitHub
      3. Commit the fixed file content
      4. Open a Fix PR targeting the original PR's base branch
      5. Mark patch status=APPLIED in DB
    """
    async with get_db_session() as session:
        patch_repo = FixPatchRepo(session)
        review_repo = ReviewRepo(session)

        # Load patch
        patch_model = await session.get(
            __import__("storage.models", fromlist=["FixPatchModel"]).FixPatchModel,
            uuid.UUID(patch_id),
        )
        if patch_model is None:
            raise HTTPException(status_code=404, detail="Patch not found")

        if patch_model.status != FixStatus.APPROVED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Patch must be APPROVED before applying. Current status: {patch_model.status}",
            )

        # Load review context
        review = await review_repo.get_by_id(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")

        pr = review.pull_request
        repo_full_name = pr.repository.full_name
        base_branch = pr.base_branch
        pr_number = pr.pr_number

        # Apply the unified diff to get fixed content
        fixed_content = _apply_unified_diff(patch_model.unified_diff, patch_model.file_path)
        if fixed_content is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to apply unified diff to produce fixed content",
            )

        # Create GitHub fix branch + commit + PR
        github_client = GitHubClient()
        fix_branch = generate_fix_branch_name(pr_number, patch_id)
        commit_msg = fix_commit_message(patch_model.file_path, patch_model.explanation)

        try:
            await github_client.create_fix_branch(
                repo_full_name=repo_full_name,
                base_sha=pr.head_sha,
                branch_name=fix_branch,
            )
            await github_client.commit_fix(
                repo_full_name=repo_full_name,
                branch_name=fix_branch,
                file_path=patch_model.file_path,
                new_content=fixed_content,
                commit_message=commit_msg,
            )
            fix_pr_url, fix_pr_number = await github_client.open_fix_pr(
                repo_full_name=repo_full_name,
                fix_branch=fix_branch,
                base_branch=base_branch,
                pr_number=pr_number,
                findings_summary=patch_model.explanation,
            )
        except Exception as exc:
            logger.error("Failed to apply fix to GitHub", error=str(exc))
            raise HTTPException(
                status_code=500,
                detail=f"GitHub operation failed: {exc}",
            )

        # Mark as applied in DB
        await patch_repo.mark_applied(
            patch_id=patch_id,
            fix_branch=fix_branch,
            fix_pr_url=fix_pr_url,
            fix_pr_number=fix_pr_number,
        )

        logger.info(
            "Fix applied",
            patch_id=patch_id,
            fix_branch=fix_branch,
            fix_pr_url=fix_pr_url,
        )
        return {
            "status": "applied",
            "fix_branch": fix_branch,
            "fix_pr_url": fix_pr_url,
            "fix_pr_number": fix_pr_number,
        }


def _apply_unified_diff(unified_diff: str, file_path: str) -> str | None:
    """
    Extract the 'after' version of a file from a unified diff.

    Since we store the full fixed_content in LLM responses, we reconstruct
    by replaying the diff additions. For production, use `patch` subprocess.
    """
    lines = unified_diff.splitlines()
    result_lines = []
    in_hunk = False

    for line in lines:
        if line.startswith("---") or line.startswith("+++"):
            in_hunk = True
            continue
        if line.startswith("@@"):
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            result_lines.append(line[1:])
        elif line.startswith("-"):
            continue  # Skip removed lines
        else:
            result_lines.append(line[1:] if line.startswith(" ") else line)

    return "\n".join(result_lines) if result_lines else None