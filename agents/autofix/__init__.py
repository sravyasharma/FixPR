"""
agents/autofix/__init__.py

Auto-Fix Agent — generates code patches for HIGH/CRITICAL findings.

Workflow per finding:
  1. Read current file content from local repo clone
  2. Call GPT-4o with finding + file context → patched file content
  3. Compute unified diff (original vs patched)
  4. Validate patch (apply it, run linting, optionally run tests)
  5. Return PatchSchema — stored with status=PENDING_APPROVAL

After human approval (separate endpoint):
  6. Create fix branch in GitHub
  7. Commit patched file
  8. Open Fix PR targeting base branch

Architecture role:
  - Called by orchestrator node_autofix after merge
  - Human approval handled by webhook_api approval endpoint
  - Fix PR creation handled by github_client.py
"""

from __future__ import annotations

import difflib
import json
import subprocess
import time
from pathlib import Path

from openai import AsyncOpenAI

from shared.config import get_settings
from shared.constants import AnalysisSource, FixStatus
from shared.logger import get_logger
from shared.schemas import FindingSchema, PatchSchema

logger = get_logger(__name__)

AUTOFIX_SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing a specific code issue.

You will be given:
1. The current file content
2. A specific finding (bug/security issue) with location and description
3. A suggested fix

Your task:
- Fix ONLY the specific issue described
- Make the minimal change necessary
- Do NOT refactor unrelated code
- Do NOT change formatting or style unless directly related to the fix
- Preserve all comments, docstrings, and surrounding logic

Respond with a JSON object containing exactly:
{
  "fixed_content": "<complete file content with the fix applied>",
  "explanation": "<one or two sentences explaining what was changed and why>"
}

Respond ONLY with valid JSON. No markdown fences, no preamble."""


class AutoFixAgent:
    """
    Generates code patches for individual findings.

    Uses GPT-4o to produce a fixed version of the file, then computes
    a unified diff between original and fixed content.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

    async def generate_fix(
        self,
        finding: FindingSchema,
        repo_path: str,
        repo_full_name: str,
        pr_number: int,
        base_branch: str,
    ) -> PatchSchema | None:
        """
        Generate a patch for a single finding.

        Returns PatchSchema on success, None if the fix cannot be generated.
        """
        if not repo_path:
            logger.warning("No repo path provided for auto-fix")
            return None

        file_abs_path = Path(repo_path) / finding.file_path
        if not file_abs_path.exists():
            logger.warning(
                "File not found for auto-fix",
                file=finding.file_path,
                repo_path=repo_path,
            )
            return None

        # Read original content
        try:
            original_content = file_abs_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to read file for auto-fix", error=str(exc))
            return None

        logger.info(
            "Generating auto-fix",
            file=finding.file_path,
            line=finding.line_number,
            severity=finding.severity,
        )

        # 1. Generate fix via LLM
        fixed_content, explanation = await self._generate_with_llm(
            original_content=original_content,
            finding=finding,
        )

        if fixed_content is None:
            return None

        # 2. Compute unified diff
        unified_diff = self._compute_diff(
            original=original_content,
            fixed=fixed_content,
            file_path=finding.file_path,
        )

        if not unified_diff.strip():
            logger.warning(
                "LLM produced no diff — fix may be a no-op",
                file=finding.file_path,
            )
            return None

        # 3. Validate the patch
        tests_passed, lint_passed, validation_errors = await self._validate_patch(
            repo_path=repo_path,
            file_path=finding.file_path,
            fixed_content=fixed_content,
            original_content=original_content,
        )

        patch = PatchSchema(
            finding_id=finding.id,
            file_path=finding.file_path,
            unified_diff=unified_diff,
            explanation=explanation,
            tests_passed=tests_passed,
            lint_passed=lint_passed,
            validation_errors=validation_errors,
            status=FixStatus.GENERATED,
        )

        logger.info(
            "Auto-fix generated",
            file=finding.file_path,
            patch_id=patch.id,
            tests_passed=tests_passed,
            lint_passed=lint_passed,
            diff_lines=len(unified_diff.splitlines()),
        )

        return patch

    async def _generate_with_llm(
        self,
        original_content: str,
        finding: FindingSchema,
    ) -> tuple[str | None, str]:
        """Call GPT-4o to produce fixed file content."""
        # Truncate very large files to stay within context
        MAX_FILE_CHARS = 40_000
        truncated = original_content[:MAX_FILE_CHARS]
        if len(original_content) > MAX_FILE_CHARS:
            truncated += "\n... [file truncated for context limit]"

        # Highlight the problematic area
        context_lines = self._extract_context(original_content, finding.line_number)

        user_prompt = f"""FILE: {finding.file_path}

FINDING:
- Severity: {finding.severity}
- Line: {finding.line_number or 'N/A'}
- Issue: {finding.issue}
- Suggestion: {finding.suggestion}

PROBLEMATIC CODE CONTEXT (around line {finding.line_number or 'N/A'}):
```
{context_lines}
```

FULL FILE CONTENT:
```
{truncated}
```

Fix the specific issue described above. Return the complete fixed file content."""

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.openai_model,
                temperature=0.0,  # Deterministic for code fixes
                max_tokens=self._settings.openai_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": AUTOFIX_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            logger.error("LLM auto-fix call failed", error=str(exc))
            return None, ""

        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            fixed_content = parsed.get("fixed_content", "")
            explanation = parsed.get("explanation", "AI-generated fix")
            if not fixed_content:
                return None, ""
            return fixed_content, explanation
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM fix response", error=str(exc))
            return None, ""

    def _compute_diff(
        self, original: str, fixed: str, file_path: str
    ) -> str:
        """Generate a unified diff between original and fixed content."""
        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        return "".join(diff)

    async def _validate_patch(
        self,
        repo_path: str,
        file_path: str,
        fixed_content: str,
        original_content: str,
    ) -> tuple[bool | None, bool | None, list[str]]:
        """
        Validate the generated patch.

        Steps:
        1. Write fixed content to disk temporarily
        2. Run linting (if enabled)
        3. Run tests (if enabled)
        4. Restore original content regardless of outcome
        """
        settings = self._settings
        abs_path = Path(repo_path) / file_path
        validation_errors: list[str] = []
        tests_passed: bool | None = None
        lint_passed: bool | None = None

        # Write fixed content
        try:
            abs_path.write_text(fixed_content, encoding="utf-8")
        except Exception as exc:
            validation_errors.append(f"Could not write fixed content: {exc}")
            return None, None, validation_errors

        try:
            # Run linting
            if settings.autofix_run_lint:
                lint_passed, lint_errors = self._run_lint(repo_path, file_path)
                validation_errors.extend(lint_errors)

            # Run tests
            if settings.autofix_run_tests:
                tests_passed, test_errors = self._run_tests(repo_path)
                validation_errors.extend(test_errors)

        finally:
            # Always restore original content
            try:
                abs_path.write_text(original_content, encoding="utf-8")
            except Exception as exc:
                logger.error("Failed to restore original file content", error=str(exc))

        return tests_passed, lint_passed, validation_errors

    def _run_lint(
        self, repo_path: str, file_path: str
    ) -> tuple[bool, list[str]]:
        """Run flake8 on the patched file. Returns (passed, errors)."""
        abs_path = str(Path(repo_path) / file_path)
        try:
            result = subprocess.run(
                ["python", "-m", "flake8", "--max-line-length=120", abs_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=repo_path,
            )
            if result.returncode != 0:
                errors = [
                    line for line in result.stdout.splitlines()
                    if line.strip() and not any(
                        # Ignore pure style issues in auto-fix validation
                        code in line for code in ["E1", "E2", "E3", "W3", "W5"]
                    )
                ]
                return len(errors) == 0, errors[:10]
            return True, []
        except Exception as exc:
            return None, [f"Lint check error: {exc}"]

    def _run_tests(self, repo_path: str) -> tuple[bool | None, list[str]]:
        """Run pytest in the repo. Returns (passed, errors)."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--tb=short", "-q", "--timeout=30"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=repo_path,
            )
            if result.returncode == 5:
                # pytest exit code 5 = no tests collected — treat as pass
                return True, []
            if result.returncode != 0:
                # Capture first 10 lines of failure output
                error_lines = result.stdout.splitlines()[:10]
                return False, error_lines
            return True, []
        except subprocess.TimeoutExpired:
            return None, ["Test suite timed out (120s)"]
        except Exception as exc:
            return None, [f"Test run error: {exc}"]

    def _extract_context(
        self, content: str, line_number: int | None, context_lines: int = 10
    ) -> str:
        """Extract lines around the finding for LLM context."""
        if line_number is None:
            return content[:2000]

        lines = content.splitlines()
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)

        numbered = []
        for i, line in enumerate(lines[start:end], start=start + 1):
            marker = ">>>" if i == line_number else "   "
            numbered.append(f"{marker} {i:4d} | {line}")
        return "\n".join(numbered)