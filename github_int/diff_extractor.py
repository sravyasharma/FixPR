"""
github/diff_extractor.py

Extract changed files and unified diffs from a PR using git diff output.

Architecture role:
  - Called after clone_repo.py checks out the PR head SHA
  - Produces PRDiffSchema - the input to all three analysis agents
  - Infers file language from extension for routing to correct analyzers
"""

from __future__ import annotations

from pathlib import Path

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

    Compares the base SHA against the PR head SHA using raw git diff output.
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
        Compute the diff between the PR base and head commits.

        Returns PRDiffSchema with all changed files and their patches.
        """
        repo = Repo(local_repo_path)
        changed_files: list[FileChangeSchema] = []
        skipped = 0

        name_only_output = repo.git.diff(base_sha, head_sha, "--name-only")
        changed_file_paths = [
            file_path
            for file_path in name_only_output.splitlines()
            if file_path
        ]

        for file_path in changed_file_paths:
            status_output = repo.git.diff(
                base_sha,
                head_sha,
                "--name-status",
                "--",
                file_path,
            )
            status_code = status_output.split(maxsplit=1)[0] if status_output else "M"
            change_type = _map_change_type(status_code[:1])

            raw_patch = repo.git.diff(base_sha, head_sha, "--", file_path)
            patch_size = len(raw_patch.encode("utf-8"))

            if patch_size > MAX_PATCH_SIZE_BYTES:
                logger.debug(
                    "Skipping large diff",
                    file=file_path,
                    size=patch_size,
                )
                patch_text = f"[Diff truncated - {patch_size} bytes]"
                skipped += 1
            elif raw_patch:
                patch_text = raw_patch
            else:
                patch_text = None

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
                    change_type=change_type,
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

        diff = PRDiffSchema(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            head_sha=head_sha,
            base_sha=base_sha,
            local_repo_path=local_repo_path,
            changed_files=changed_files,
        )
        repo.close()
        return diff
