"""
github/github_client.py

GitHub REST API client for the AI Code Review Platform.

Responsibilities:
  - Post PR review comments (line-level and summary)
  - Create fix branches
  - Commit and push patches
  - Open Fix PRs
  - Fetch PR metadata

Uses PyGithub for authenticated API calls.
All methods are async-safe (PyGithub is sync — run in executor).

Architecture role:
  - Called by the worker AFTER the pipeline completes
  - Called by the auto-fix workflow to create Fix PRs
  - Never called from the webhook API (that would block)
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from github import Github, GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

from shared.config import get_settings
from shared.constants import Severity, SeverityScore
from shared.logger import get_logger
from shared.schemas import FindingSchema, MergedReviewResult

logger = get_logger(__name__)

# Emoji badges for each severity in the posted review
SEVERITY_EMOJI: dict[str, str] = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
    Severity.INFO: "⚪",
}


@lru_cache(maxsize=1)
def _get_github_client() -> Github:
    settings = get_settings()
    return Github(settings.github_token)


def _get_repo(full_name: str) -> Repository:
    """Return a PyGithub Repository object."""
    return _get_github_client().get_repo(full_name)


class GitHubClient:
    """
    Async-friendly GitHub API client.

    PyGithub is synchronous. We run all blocking calls in
    asyncio.get_event_loop().run_in_executor(None, ...) so the
    async worker event loop is never blocked.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._loop = None

    async def _run_sync(self, func, *args, **kwargs):
        """Execute a synchronous callable in the default thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # ---------------------------------------------------------------- #
    # Review Comments
    # ---------------------------------------------------------------- #

    async def post_review(
        self,
        repo_full_name: str,
        pr_number: int,
        result: MergedReviewResult,
        head_sha: str,
    ) -> None:
        """
        Post the full review as a GitHub PR Review.

        Strategies:
        1. Create a review with COMMENT event (non-blocking for PR author)
        2. Post individual line-level comments for findings with line numbers
        3. Post a summary comment with the full markdown report
        """
        settings = self._settings
        if not settings.post_review_comments:
            logger.info("Review posting disabled — skipping")
            return

        try:
            await self._run_sync(
                self._post_review_sync,
                repo_full_name,
                pr_number,
                result,
                head_sha,
            )
        except Exception as exc:
            logger.error("Failed to post GitHub review", error=str(exc))
            raise

    def _post_review_sync(
        self,
        repo_full_name: str,
        pr_number: int,
        result: MergedReviewResult,
        head_sha: str,
    ) -> None:
        settings = get_settings()
        repo = _get_repo(repo_full_name)
        pr: PullRequest = repo.get_pull(pr_number)

        if not hasattr(pr, "create_review"):
            raise RuntimeError("GitHub pull request object does not support review creation")

        # Build line-level review comments (capped to avoid spam)
        review_comments = []
        findings_with_lines = [
            f for f in result.findings if f.line_number is not None
        ]
        capped = findings_with_lines[: settings.max_comments_per_review]

        for finding in capped:
            body = self._format_finding_comment(finding)
            review_comments.append(
                {
                    "path": finding.file_path,
                    "line": finding.line_number,
                    "body": body,
                }
            )

        # Create the review
        try:
            pr.create_review(
                commit=repo.get_commit(head_sha),
                body=result.markdown_report,
                event="COMMENT",
                comments=review_comments,
            )
            logger.info(
                "GitHub review posted",
                pr_number=pr_number,
                repo=repo_full_name,
                comment_count=len(review_comments),
            )
        except GithubException as exc:
            logger.error(
                "Failed to post GitHub review",
                pr_number=pr_number,
                repo=repo_full_name,
                error=str(exc),
            )
            # Fallback: post as a simple PR comment
            pr.create_issue_comment(result.markdown_report[:65536])

    def _format_finding_comment(self, finding: FindingSchema) -> str:
        emoji = SEVERITY_EMOJI.get(finding.severity, "⚪")
        lines = [
            f"{emoji} **[{finding.severity}]** {finding.issue}",
            f"",
            f"**Source:** `{finding.source}`",
        ]
        if finding.rule_id:
            lines.append(f"**Rule:** `{finding.rule_id}`")
        lines += [
            f"",
            f"**Suggestion:** {finding.suggestion}",
        ]
        if finding.code_snippet:
            lines += [f"", f"```", finding.code_snippet, f"```"]
        return "\n".join(lines)

    # ---------------------------------------------------------------- #
    # Branch Management
    # ---------------------------------------------------------------- #

    async def create_fix_branch(
        self, repo_full_name: str, base_sha: str, branch_name: str
    ) -> str:
        """Create a new branch from base_sha. Returns the branch ref."""
        return await self._run_sync(
            self._create_fix_branch_sync, repo_full_name, base_sha, branch_name
        )

    def _create_fix_branch_sync(
        self, repo_full_name: str, base_sha: str, branch_name: str
    ) -> str:
        repo = _get_repo(repo_full_name)
        ref = repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_sha)
        logger.info(
            "Fix branch created",
            repo=repo_full_name,
            branch=branch_name,
            sha=base_sha,
        )
        return ref.ref

    # ---------------------------------------------------------------- #
    # Commit & Push Patches
    # ---------------------------------------------------------------- #

    async def commit_fix(
        self,
        repo_full_name: str,
        branch_name: str,
        file_path: str,
        new_content: str,
        commit_message: str,
    ) -> str:
        """Update a file on the fix branch. Returns new commit SHA."""
        return await self._run_sync(
            self._commit_fix_sync,
            repo_full_name,
            branch_name,
            file_path,
            new_content,
            commit_message,
        )

    def _commit_fix_sync(
        self,
        repo_full_name: str,
        branch_name: str,
        file_path: str,
        new_content: str,
        commit_message: str,
    ) -> str:
        repo = _get_repo(repo_full_name)
        try:
            # Get current file to get its SHA (required for update)
            contents = repo.get_contents(file_path, ref=branch_name)
            result = repo.update_file(
                path=file_path,
                message=commit_message,
                content=new_content,
                sha=contents.sha,
                branch=branch_name,
            )
        except GithubException:
            # File doesn't exist yet — create it
            result = repo.create_file(
                path=file_path,
                message=commit_message,
                content=new_content,
                branch=branch_name,
            )
        sha = result["commit"].sha
        logger.info(
            "Fix committed",
            repo=repo_full_name,
            branch=branch_name,
            file=file_path,
            sha=sha,
        )
        return sha

    # ---------------------------------------------------------------- #
    # Open Fix PR
    # ---------------------------------------------------------------- #

    async def open_fix_pr(
        self,
        repo_full_name: str,
        fix_branch: str,
        base_branch: str,
        pr_number: int,
        findings_summary: str,
    ) -> tuple[str, int]:
        """
        Open a Fix PR from fix_branch → base_branch.

        Returns (html_url, pr_number).
        """
        return await self._run_sync(
            self._open_fix_pr_sync,
            repo_full_name,
            fix_branch,
            base_branch,
            pr_number,
            findings_summary,
        )

    def _open_fix_pr_sync(
        self,
        repo_full_name: str,
        fix_branch: str,
        base_branch: str,
        pr_number: int,
        findings_summary: str,
    ) -> tuple[str, int]:
        repo = _get_repo(repo_full_name)
        title = f"🤖 Auto-fix: Issues from PR #{pr_number}"
        body = (
            f"This PR was automatically generated by the AI Code Review Platform.\n\n"
            f"It addresses findings from PR #{pr_number}.\n\n"
            f"## Addressed Findings\n\n{findings_summary}\n\n"
            f"---\n"
            f"⚠️ **Please review carefully before merging.** "
            f"These fixes are AI-generated and require human verification."
        )
        pr = repo.create_pull(
            title=title,
            body=body,
            head=fix_branch,
            base=base_branch,
            draft=False,
        )
        logger.info(
            "Fix PR opened",
            repo=repo_full_name,
            fix_branch=fix_branch,
            fix_pr_number=pr.number,
            fix_pr_url=pr.html_url,
        )
        return pr.html_url, pr.number

    # ---------------------------------------------------------------- #
    # PR Info
    # ---------------------------------------------------------------- #

    async def get_pr_files(self, repo_full_name: str, pr_number: int) -> list[str]:
        """Return list of file paths changed in the PR."""
        return await self._run_sync(
            self._get_pr_files_sync, repo_full_name, pr_number
        )

    def _get_pr_files_sync(self, repo_full_name: str, pr_number: int) -> list[str]:
        repo = _get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        return [f.filename for f in pr.get_files()]