"""
agents/static_analysis/mypy_analyzer.py

MyPy type checking analyzer for Python files.

MyPy is particularly valuable for catching:
- Type mismatches before runtime
- Missing None checks
- Incorrect argument types
- Return type violations

Uses --ignore-missing-imports to prevent false positives from
third-party packages without stubs.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from shared.constants import AnalysisSource, ConfidenceLevel, Severity
from shared.logger import get_logger
from shared.schemas import FileChangeSchema, FindingSchema

logger = get_logger(__name__)

MYPY_ERROR_CODES_HIGH: frozenset[str] = frozenset({
    "assignment",
    "arg-type",
    "return-value",
    "return",
    "attr-defined",
    "call-arg",
    "call-overload",
    "no-untyped-call",
    "type-arg",
    "valid-type",
    "name-defined",
    "index",
    "operator",
})


def _severity_for_mypy(error_code: str, severity_str: str) -> Severity:
    """Map mypy error codes to our severity levels."""
    if severity_str == "error" and error_code in MYPY_ERROR_CODES_HIGH:
        return Severity.HIGH
    if severity_str == "error":
        return Severity.MEDIUM
    if severity_str == "warning":
        return Severity.LOW
    return Severity.INFO


class AnalyzerResult:
    def __init__(self):
        self.findings: list[FindingSchema] = []
        self.files_analyzed: list[str] = []


class MypyAnalyzer:
    """Runs mypy and normalizes type error findings."""

    async def analyze(
        self, repo_path: str, files: list[FileChangeSchema]
    ) -> AnalyzerResult:
        result = AnalyzerResult()
        if not files:
            return result

        file_paths = [
            str(Path(repo_path) / f.file_path)
            for f in files
            if (Path(repo_path) / f.file_path).exists()
        ]
        result.files_analyzed = [f.file_path for f in files]

        if not file_paths:
            return result

        cmd = [
            "python", "-m", "mypy",
            "--output=json",
            "--ignore-missing-imports",
            "--no-error-summary",
            "--no-pretty",
            "--show-error-codes",
            *file_paths,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path,
                env={**os.environ, "PYTHONPATH": repo_path},
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            logger.error("MyPy timed out")
            return result
        except Exception as exc:
            logger.error("MyPy subprocess failed", error=str(exc))
            return result

        raw = stdout.decode("utf-8", errors="replace").strip()
        if not raw:
            return result

        # MyPy JSON output: one JSON object per line
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # MyPy may emit non-JSON lines for notes
                continue

            severity_str = entry.get("severity", "error")
            if severity_str == "note":
                continue  # Skip "note" context lines

            error_code = entry.get("code", "")
            file_str = entry.get("file", "")
            rel_path = file_str.replace(repo_path.rstrip("/") + "/", "")

            finding = FindingSchema(
                source=AnalysisSource.MYPY,
                rule_id=error_code or None,
                file_path=rel_path or file_str,
                line_number=entry.get("line"),
                column=entry.get("column"),
                severity=_severity_for_mypy(error_code, severity_str),
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.88,
                issue=f"[mypy:{error_code}] {entry.get('message', '')}",
                suggestion=self._make_suggestion(error_code, entry.get("message", "")),
            )
            result.findings.append(finding)

        logger.debug("MyPy complete", findings=len(result.findings))
        return result

    def _make_suggestion(self, code: str, message: str) -> str:
        suggestions: dict[str, str] = {
            "assignment": "Ensure the assigned value matches the declared type",
            "arg-type": "Pass an argument matching the expected parameter type",
            "return-value": "Return a value matching the declared return type",
            "attr-defined": "Verify the attribute exists on the object or add a type guard",
            "no-untyped-call": "Add type annotations to the called function",
        }
        return suggestions.get(code, f"Fix type error: {message[:120]}")