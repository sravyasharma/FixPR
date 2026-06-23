"""
github/diff_extractor.py

Extract changed files and unified diffs from a PR using GitPython.

Architecture role:
  - Called after clone_repo.py checks out the PR head SHA
  - Produces PRDiffSchema — the input to all three analysis agents
  - Infers file language from extension for routing to correct analyzers
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

import git
from git import Repo

from shared.constants import (
    JS_TS_EXTENSIONS,
    PYTHON_EXTENSIONS,
)
from shared.logger import get_logger
from shared.schemas import FileChangeSchema, PRDiffSchema

logger = get_logger(__name__)

# Maximum patch size per file to include in the diff (bytes)
# Prevents feeding enormous generated files to LLMs
MAX_PATCH_SIZE_BYTES: int = 50_000


def _infer_language(file_path: str) -> str | None:
    """Infer programming language from file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix in PYTHON_EXTENSIONS:
        return "python"
    if suffix in JS_TS_EXTENSIONS:
        return "javascript" if suffix in {".js", ".jsx", ".mjs", ".cjs"} else "typescript"
    if suffix == ".go":
        return "go"
    if suffix in {".java"}:
        return "java"
    if suffix in {".rb"}:
        return "ruby"
    if suffix in {".rs"}:
        return "rust"
    if suffix in {".sh", ".bash"}:
        return "shell"
    if suffix in {".yml", ".yaml"}:
        return "yaml"
    if suffix == ".json":
        return "json"
    if suffix in {".tf", ".tfvars"}:
        return "terraform"
    if suffix in {".dockerfile"} or Path(file_path).name.lower() == "dockerfile":
        return "dockerfile"
    return None


def _map_change_type(change_type: str) -> str:
    """Normalize git change type char to human-readable label."""
    mapping = {
        "A": "added",
        "M": "modified",
        "D": "deleted",
        "R": "renamed",
        "C": "copied",
        "T": "type_changed",
        "U": "unmerged",
    }
    return mapping.get(change_type.upper(), "modified")


class DiffExtractor:
    """
    Extracts PR diff information from a locally cloned repository.

    Compares HEAD (PR head SHA) against the merge base with the base branch,
    which gives us exactly the files the PR author changed — not all diverged commits.
    """

    def extract(
        self,
        local_repo_path: str,
        repo_full_name: str,
        pr_number: int,
        head_sha: str,
        base_sha: str,
        base_branch: str = "main",
    ) -> PRDiffSchema:
        """
        Compute the diff between the PR head and its merge base.

        Returns PRDiffSchema with all changed files and their patches.
        """
        repo = Repo(local_repo_path)

        # Find the merge base — the point where PR branch diverged from base
        try:
            merge_base_commits = repo.merge_base(head_sha, base_sha)
            if merge_base_commits:
                merge_base = merge_base_commits[0]
            else:
                # Fallback: diff directly against base_sha
                merge_base = repo.commit(base_sha)
        except Exception as exc:
            logger.warning(
                "Could not compute merge base, using base_sha directly",
                error=str(exc),
                head_sha=head_sha,
                base_sha=base_sha,
            )
            merge_base = repo.commit(base_sha)

        head_commit = repo.commit(head_sha)

        # Get diff between merge base and PR head
        diffs = merge_base.diff(head_commit, create_patch=True)

        changed_files: list[FileChangeSchema] = []
        skipped = 0

        for diff_item in diffs:
            # Determine file path (b_path for new name, handles renames)
            file_path: str = diff_item.b_path or diff_item.a_path

            # Skip binary files and very large files
            if diff_item.diff and len(diff_item.diff) > MAX_PATCH_SIZE_BYTES:
                logger.debug(
                    "Skipping large diff",
                    file=file_path,
                    size=len(diff_item.diff),
                )
                patch_text = f"[Diff truncated — {len(diff_item.diff)} bytes]"
                skipped += 1
            elif diff_item.diff:
                try:
                    patch_text = diff_item.diff.decode("utf-8", errors="replace")
                except Exception:
                    patch_text = None
            else:
                patch_text = None

            # Count additions/deletions from patch
            additions = 0
            deletions = 0
            if patch_text and not patch_text.startswith("[Diff"):
                for line in patch_text.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        additions += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        deletions += 1

            changed_files.append(
                FileChangeSchema(
                    file_path=file_path,
                    change_type=_map_change_type(diff_item.change_type),
                    patch=patch_text,
                    additions=additions,
                    deletions=deletions,
                    language=_infer_language(file_path),
                )
            )

        logger.info(
            "Diff extracted",
            repo=repo_full_name,
            pr_number=pr_number,
            changed_files=len(changed_files),
            skipped_large=skipped,
        )

        return PRDiffSchema(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
            base_sha=base_sha,
            local_repo_path=local_repo_path,
            changed_files=changed_files,
        )