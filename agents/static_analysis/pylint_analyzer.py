"""
agents/static_analysis/pylint_analyzer.py

Pylint analyzer — runs pylint on changed Python files and normalizes output.

Pylint is invoked as a subprocess (not imported) to:
1. Avoid global state pollution from pylint's module system
2. Ensure each analysis runs in a clean environment
3. Allow parallel execution without GIL contention

Output format: JSON (--output-format=json)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from shared.constants import AnalysisSource, ConfidenceLevel, Severity
from shared.logger import get_logger
from shared.schemas import FileChangeSchema, FindingSchema

logger = get_logger(__name__)


# Pylint message category → Severity mapping
PYLINT_SEVERITY_MAP: dict[str, Severity] = {
    "E": Severity.HIGH,       # Error
    "F": Severity.CRITICAL,   # Fatal
    "W": Severity.MEDIUM,     # Warning
    "C": Severity.LOW,        # Convention
    "R": Severity.INFO,       # Refactor
    "I": Severity.INFO,       # Info
}

# Rules that are style-only — skip these (handled by flake8)
STYLE_ONLY_RULES: frozenset[str] = frozenset({
    "C0103",  # invalid-name
    "C0114",  # missing-module-docstring
    "C0115",  # missing-class-docstring
    "C0116",  # missing-function-docstring
    "C0301",  # line-too-long
    "C0303",  # trailing-whitespace
    "C0304",  # final-newline
    "C0305",  # trailing-newlines
    "C0321",  # multiple-statements
    "W0611",  # unused-import (flake8 F401 covers this)
})


class AnalyzerResult:
    """Internal result container."""
    def __init__(self):
        self.findings: list[FindingSchema] = []
        self.files_analyzed: list[str] = []


class PylintAnalyzer:
    """Runs pylint and normalizes findings."""

    async def analyze(
        self, repo_path: str, files: list[FileChangeSchema]
    ) -> AnalyzerResult:
        """
        Run pylint on the given Python files.

        Runs as a subprocess with JSON output format.
        Filters out style-only rules (those belong to flake8).
        """
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
            "python", "-m", "pylint",
            "--output-format=json",
            "--score=no",
            "--disable=all",
            "--enable=E,F,W,C,R",  # All categories, we filter below
            "--max-line-length=120",
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
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            logger.error("Pylint timed out", repo_path=repo_path)
            return result
        except Exception as exc:
            logger.error("Pylint subprocess failed", error=str(exc))
            return result

        # Pylint returns exit code 0 (no issues), 1+ (issues found), 2 (fatal)
        # All are valid — parse output regardless
        raw = stdout.decode("utf-8", errors="replace").strip()
        if not raw:
            return result

        try:
            messages: list[dict[str, Any]] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse pylint JSON output", raw=raw[:500])
            return result

        for msg in messages:
            msg_id: str = msg.get("message-id", "")

            # Skip style-only rules — keep findings substantive
            if msg_id in STYLE_ONLY_RULES:
                continue

            category: str = msg.get("type", "W")[0].upper()
            severity = PYLINT_SEVERITY_MAP.get(category, Severity.LOW)

            # Convert absolute path back to relative
            abs_path: str = msg.get("path", "")
            rel_path = abs_path.replace(repo_path.rstrip("/") + "/", "") if abs_path else ""

            finding = FindingSchema(
                source=AnalysisSource.PYLINT,
                rule_id=msg_id,
                file_path=rel_path or msg.get("path", "unknown"),
                line_number=msg.get("line"),
                column=msg.get("column"),
                severity=severity,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.85,
                issue=f"[{msg_id}] {msg.get('message', '')}",
                suggestion=self._make_suggestion(msg_id, msg.get("message", "")),
            )
            result.findings.append(finding)

        logger.debug("Pylint complete", findings=len(result.findings))
        return result

    def _make_suggestion(self, rule_id: str, message: str) -> str:
        """Generate a minimal suggestion string from the pylint message."""
        suggestions: dict[str, str] = {
            "E0001": "Fix the syntax error indicated in the message",
            "E0102": "Rename the duplicate definition",
            "E0401": "Ensure the import is installed and on PYTHONPATH",
            "E1101": "Check that the attribute exists on the type",
            "W0612": "Remove the unused variable or use _ as the name",
            "W0613": "Remove unused argument or prefix with underscore",
            "W0703": "Catch a more specific exception instead of Exception",
        }
        return suggestions.get(rule_id, f"Review pylint rule {rule_id}: {message[:100]}")