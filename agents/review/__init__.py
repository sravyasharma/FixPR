"""
agents/review/__init__.py

LLM Review Agent — GPT-4o powered code review with structured output.

Reviews for:
  - Logic bugs and off-by-one errors
  - Edge cases (null/empty/negative inputs, integer overflow, etc.)
  - Performance issues (N+1 queries, unnecessary copies, unbounded loops)
  - Architectural concerns (tight coupling, SRP violations, abstraction leaks)
  - Security concerns (injection, auth bypass, IDOR, unsafe deserialization)

Explicitly EXCLUDES:
  - Style issues (handled by static analyzers)
  - Formatting (handled by static analyzers)

Architecture role:
  - Runs concurrently with StaticAnalysisAgent and SecurityAnalysisAgent
  - Uses OpenAI structured output (response_format=json_schema) for reliable parsing
  - Tracks token usage and cost for every call
  - Processes files in batches to stay within context limits
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from shared.config import get_settings
from shared.constants import (
    MODEL_COST_MAP,
    AnalysisSource,
    ConfidenceLevel,
    Severity,
)
from shared.logger import get_logger
from shared.schemas import (
    FileChangeSchema,
    FindingSchema,
    LLMReviewResult,
    PRDiffSchema,
)

logger = get_logger(__name__)

# Maximum characters of diff to include per LLM call
MAX_DIFF_CHARS_PER_CALL: int = 30_000

# Maximum files per LLM batch (to stay within context window)
MAX_FILES_PER_BATCH: int = 5

# Only include files with actual code changes
REVIEWABLE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go",
    ".java", ".rb", ".rs", ".cpp", ".c", ".h", ".cs",
    ".php", ".swift", ".kt", ".scala", ".md",
})

REVIEW_SYSTEM_PROMPT = """You are a senior software engineer performing a rigorous code review.

Your task is to analyze the provided code diff and identify:
1. LOGIC BUGS — Incorrect conditionals, off-by-one errors, wrong operator precedence
2. EDGE CASES — Unhandled null/None/empty inputs, negative numbers, empty collections, overflow
3. PERFORMANCE — N+1 queries, unnecessary loops, large memory allocations, O(n²) where O(n) is possible
4. ARCHITECTURE — Tight coupling, SRP violations, abstraction leaks, missing error propagation
5. SECURITY — SQL injection, XSS, SSRF, auth bypass, insecure deserialization, path traversal

DO NOT report:
- Formatting or indentation issues
- Naming conventions
- Missing docstrings or comments
- Code style preferences

For each issue found, you must respond with a JSON array of findings.
If no issues are found, return an empty array [].

Each finding must have exactly these fields:
{
  "file_path": "path/to/file.py",
  "line_number": 42,
  "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
  "confidence": "HIGH|MEDIUM|LOW",
  "issue": "One-sentence description of the problem",
  "suggestion": "Specific, actionable fix recommendation",
  "code_snippet": "Optional: the problematic code (1-3 lines)"
}

Respond ONLY with a valid JSON array. No preamble, no explanation, no markdown fences."""


SEVERITY_MAP: dict[str, Severity] = {
    "CRITICAL": Severity.CRITICAL,
    "HIGH": Severity.HIGH,
    "MEDIUM": Severity.MEDIUM,
    "LOW": Severity.LOW,
    "INFO": Severity.INFO,
}

CONFIDENCE_MAP: dict[str, tuple[ConfidenceLevel, float]] = {
    "HIGH": (ConfidenceLevel.HIGH, 0.90),
    "MEDIUM": (ConfidenceLevel.MEDIUM, 0.65),
    "LOW": (ConfidenceLevel.LOW, 0.40),
}


class LLMReviewAgent:
    """
    GPT-4o powered code review agent with structured output.

    Processes changed files in batches. For each batch, constructs
    a prompt containing the diffs and sends to GPT-4o with JSON mode.

    Cost tracking: every API call records tokens and USD cost to LLMReviewResult.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

    async def review(self, diff: PRDiffSchema) -> LLMReviewResult:
        """
        Review all changed files in the PR.

        Batches files to stay within context window limits.
        Aggregates results from all batches.
        """
        start = time.monotonic()

        # Filter to reviewable code files
        reviewable = [
            f for f in diff.changed_files
            if Path(f.file_path).suffix.lower() in REVIEWABLE_EXTENSIONS
            and f.change_type != "deleted"
            and f.patch  # Must have actual changes
        ]

        if not reviewable:
            logger.info("No reviewable files in diff")
            return LLMReviewResult(duration_seconds=time.monotonic() - start)

        # Batch files to stay within context limits
        batches = self._create_batches(reviewable)

        all_findings: list[FindingSchema] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        errors: list[str] = []
        files_reviewed: list[str] = []

        for i, batch in enumerate(batches):
            logger.info(
                "Processing LLM review batch",
                batch=i + 1,
                total_batches=len(batches),
                files=[f.file_path for f in batch],
            )
            try:
                findings, prompt_tokens, completion_tokens, cost = await self._review_batch(
                    batch, diff.repo_full_name, diff.pr_number
                )
                all_findings.extend(findings)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_cost += cost
                files_reviewed.extend(f.file_path for f in batch)
            except Exception as exc:
                error_msg = f"Batch {i + 1} failed: {str(exc)}"
                errors.append(error_msg)
                logger.error("LLM batch failed", batch=i + 1, error=str(exc))

        duration = time.monotonic() - start
        logger.info(
            "LLM review complete",
            findings=len(all_findings),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cost_usd=f"${total_cost:.4f}",
            duration=f"{duration:.2f}s",
        )

        return LLMReviewResult(
            findings=all_findings,
            files_reviewed=files_reviewed,
            model_used=self._settings.openai_model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cost_usd=total_cost,
            errors=errors,
            duration_seconds=duration,
        )

    async def _review_batch(
        self,
        files: list[FileChangeSchema],
        repo_full_name: str,
        pr_number: int,
    ) -> tuple[list[FindingSchema], int, int, float]:
        """
        Review a single batch of files using GPT-4o.

        Returns: (findings, prompt_tokens, completion_tokens, cost_usd)
        """
        user_prompt = self._build_prompt(files, repo_full_name, pr_number)

        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._settings.openai_model,
            temperature=self._settings.openai_temperature,
            max_tokens=self._settings.openai_max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cost = self._calculate_cost(
            self._settings.openai_model, prompt_tokens, completion_tokens
        )

        content = response.choices[0].message.content or "[]"
        findings = self._parse_response(content, files)

        return findings, prompt_tokens, completion_tokens, cost

    def _build_prompt(
        self,
        files: list[FileChangeSchema],
        repo_full_name: str,
        pr_number: int,
    ) -> str:
        """Build the user prompt with file diffs."""
        parts = [
            f"Repository: {repo_full_name}",
            f"Pull Request: #{pr_number}",
            f"Files changed: {len(files)}",
            "",
            "CODE DIFFS TO REVIEW:",
            "=" * 60,
        ]

        total_chars = 0
        for f in files:
            file_header = f"\n## File: {f.file_path} ({f.change_type})\n"
            patch = f.patch or "[No diff available]"

            # Truncate individual patches if they're too large
            remaining_budget = MAX_DIFF_CHARS_PER_CALL - total_chars
            if len(patch) > remaining_budget:
                patch = patch[:remaining_budget] + "\n... [diff truncated]"

            diff_block = f"```diff\n{patch}\n```\n"
            total_chars += len(file_header) + len(diff_block)
            parts.extend([file_header, diff_block])

            if total_chars >= MAX_DIFF_CHARS_PER_CALL:
                parts.append("[Additional files omitted — context limit reached]")
                break

        parts.append("=" * 60)
        parts.append("\nRespond with a JSON array of findings. Return [] if no issues found.")

        return "\n".join(parts)

    def _parse_response(
        self, content: str, files: list[FileChangeSchema]
    ) -> list[FindingSchema]:
        """Parse GPT-4o JSON response into FindingSchema objects."""
        # Clean common LLM response artifacts
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            # Model returns either array or {"findings": [...]}
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                raw_findings = parsed.get("findings", parsed.get("issues", [parsed]))
            elif isinstance(parsed, list):
                raw_findings = parsed
            else:
                logger.warning("Unexpected LLM response shape", type=type(parsed).__name__)
                return []
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM JSON response", error=str(exc), content=content[:500])
            return []

        findings: list[FindingSchema] = []
        valid_file_paths = {f.file_path for f in files}

        for item in raw_findings:
            if not isinstance(item, dict):
                continue

            file_path = item.get("file_path", "")
            # Validate file is actually in the diff to prevent hallucination
            if file_path and file_path not in valid_file_paths:
                # Try to match by basename
                matching = [fp for fp in valid_file_paths if fp.endswith(file_path)]
                file_path = matching[0] if matching else file_path

            severity_str = (item.get("severity") or "MEDIUM").upper()
            confidence_str = (item.get("confidence") or "MEDIUM").upper()

            severity = SEVERITY_MAP.get(severity_str, Severity.MEDIUM)
            confidence_level, confidence_score = CONFIDENCE_MAP.get(
                confidence_str, (ConfidenceLevel.MEDIUM, 0.65)
            )

            issue = item.get("issue", "").strip()
            suggestion = item.get("suggestion", "").strip()

            if not issue or not file_path:
                continue  # Skip incomplete findings

            finding = FindingSchema(
                source=AnalysisSource.LLM_REVIEW,
                file_path=file_path,
                line_number=item.get("line_number"),
                severity=severity,
                confidence=confidence_level,
                confidence_score=confidence_score,
                issue=issue,
                suggestion=suggestion,
                code_snippet=item.get("code_snippet"),
            )
            findings.append(finding)

        return findings

    def _create_batches(self, files: list[FileChangeSchema]) -> list[list[FileChangeSchema]]:
        """Split files into batches respecting file count and size limits."""
        batches: list[list[FileChangeSchema]] = []
        current_batch: list[FileChangeSchema] = []
        current_size = 0

        for f in files:
            file_size = len(f.patch or "")
            if (len(current_batch) >= MAX_FILES_PER_BATCH or
                    current_size + file_size > MAX_DIFF_CHARS_PER_CALL) and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            current_batch.append(f)
            current_size += file_size

        if current_batch:
            batches.append(current_batch)

        return batches

    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate cost in USD based on token counts."""
        costs = MODEL_COST_MAP.get(model, {"prompt": 5.00, "completion": 15.00})
        return (
            prompt_tokens * costs["prompt"] / 1_000_000
            + completion_tokens * costs["completion"] / 1_000_000
        )