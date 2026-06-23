"""
github/clone_repo.py

Repository cloning and checkout using GitPython.

Responsibilities:
  - Clone a GitHub repo (or update an existing clone)
  - Checkout the PR's head SHA for accurate file content
  - Clean up clones after analysis (configurable)

Architecture role:
  - Called by the worker at pipeline start, before agents run
  - Returns a local filesystem path that all agents use
  - Uses shallow clone (depth=50) to minimize bandwidth
"""

from __future__ import annotations

import shutil
from pathlib import Path

import git
from git import GitCommandError, InvalidGitRepositoryError, Repo

from shared.config import get_settings
from shared.logger import get_logger
from shared.schemas import WebhookPayloadSchema

logger = get_logger(__name__)


class RepoCloner:
    """
    Clone and manage local GitHub repository copies.

    Clone path format: {base_dir}/{repo_full_name_slug}/{head_sha[:8]}
    Example: /tmp/repos/myorg--myrepo/abc12345

    Including the SHA in the path lets multiple concurrent reviews of the
    same repo proceed without conflict.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._base_dir = Path(self._settings.github_clone_base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _clone_path(self, repo_full_name: str, head_sha: str) -> Path:
        """Compute deterministic clone directory for this repo + SHA."""
        slug = repo_full_name.replace("/", "--")
        return self._base_dir / slug / head_sha[:8]

    async def clone_and_checkout(self, payload: WebhookPayloadSchema) -> str:
        """
        Clone the repository and checkout the PR head SHA.

        If a clone already exists at this path (e.g. from a previous
        run of the same SHA), reuse it — avoids redundant network I/O.

        Returns:
            Absolute path to the cloned repository on disk.
        """
        repo_url = self._authenticated_url(payload.repo_clone_url)
        clone_path = self._clone_path(payload.repo_full_name, payload.pr_head_sha)

        if clone_path.exists():
            logger.info(
                "Reusing existing clone",
                path=str(clone_path),
                sha=payload.pr_head_sha,
            )
            return str(clone_path)

        clone_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Cloning repository",
            repo=payload.repo_full_name,
            sha=payload.pr_head_sha,
            path=str(clone_path),
        )

        try:
            repo = git.Repo.clone_from(
                url=repo_url,
                to_path=str(clone_path),
                depth=50,  # Shallow clone — sufficient for diff analysis
                no_single_branch=True,  # Fetch all branches (needed for base branch)
            )

            # Checkout the exact commit the PR is at
            repo.git.checkout(payload.pr_head_sha)

            logger.info(
                "Repository cloned and checked out",
                repo=payload.repo_full_name,
                sha=payload.pr_head_sha,
                path=str(clone_path),
            )
            return str(clone_path)

        except GitCommandError as exc:
            logger.error(
                "Git clone failed",
                repo=payload.repo_full_name,
                error=str(exc),
            )
            # Clean up partial clone
            if clone_path.exists():
                shutil.rmtree(clone_path, ignore_errors=True)
            raise

    def cleanup(self, repo_full_name: str, head_sha: str) -> None:
        """
        Remove the cloned repository from disk.

        Call after the review pipeline completes to free disk space.
        Safe to call even if the path doesn't exist.
        """
        clone_path = self._clone_path(repo_full_name, head_sha)
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
            logger.info(
                "Clone cleaned up",
                repo=repo_full_name,
                sha=head_sha,
                path=str(clone_path),
            )

    def _authenticated_url(self, clone_url: str) -> str:
        """
        Inject GitHub token into the clone URL for authentication.

        Transforms:
          https://github.com/org/repo.git
          → https://x-access-token:TOKEN@github.com/org/repo.git
        """
        token = self._settings.github_token
        if clone_url.startswith("https://"):
            return clone_url.replace(
                "https://", f"https://x-access-token:{token}@", 1
            )
        return clone_url