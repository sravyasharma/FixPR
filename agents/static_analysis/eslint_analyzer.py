"""
agents/static_analysis/eslint_analyzer.py

ESLint analyzer for JavaScript and TypeScript files.

Uses ESLint's JSON formatter for structured output.
Requires eslint to be installed (npm install -g eslint or project-local).

Focuses on:
- Potential bugs and errors (not style)
- Possible security-relevant patterns
- Unused variables, undefined references
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


# ESLint severity: 1 = warn, 2 = error
ESLINT_SEVERITY_MAP: dict[int, Severity] = {
    2: Severity.HIGH,
    1: Severity.LOW,
}

# Rules that indicate potential bugs (higher confidence)
BUG_RULES: frozenset[str] = frozenset({
    "no-undef",
    "no-unused-vars",
    "no-unreachable",
    "no-constant-condition",
    "no-dupe-keys",
    "no-duplicate-case",
    "no-empty",
    "no-ex-assign",
    "no-func-assign",
    "no-sparse-arrays",
    "use-before-define",
    "no-prototype-builtins",
    "no-eval",
    "no-implied-eval",
})


class AnalyzerResult:
    def __init__(self):
        self.findings: list[FindingSchema] = []
        self.files_analyzed: list[str] = []


class ESLintAnalyzer:
    """Runs ESLint and normalizes findings."""

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

        # Prefer project-local eslint, fall back to global
        eslint_cmd = self._find_eslint(repo_path)

        cmd = [
            eslint_cmd,
            "--format=json",
            "--no-eslintrc",       # Don't load project config (isolation)
            "--env=browser,node,es2021",
            "--rule", '{"no-undef": 2, "no-unused-vars": 1, "no-unreachable": 2, '
                      '"no-constant-condition": 2, "no-dupe-keys": 2, '
                      '"no-duplicate-case": 2, "no-eval": 2, '
                      '"no-implied-eval": 2, "no-prototype-builtins": 1}',
            *file_paths,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_path,
                env={**os.environ},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=90)
        except asyncio.TimeoutError:
            logger.error("ESLint timed out")
            return result
        except FileNotFoundError:
            logger.warning("ESLint not found — skipping JS/TS analysis")
            return result
        except Exception as exc:
            logger.error("ESLint subprocess failed", error=str(exc))
            return result

        raw = stdout.decode("utf-8", errors="replace").strip()
        if not raw:
            return result

        try:
            eslint_output: list[dict[str, Any]] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse ESLint JSON output", raw=raw[:500])
            return result

        for file_result in eslint_output:
            file_abs_path: str = file_result.get("filePath", "")
            rel_path = file_abs_path.replace(repo_path.rstrip("/") + "/", "")

            for message in file_result.get("messages", []):
                rule_id: str = message.get("ruleId") or "unknown"
                eslint_severity: int = message.get("severity", 1)
                severity = ESLINT_SEVERITY_MAP.get(eslint_severity, Severity.LOW)

                # Boost severity for known bug-prone rules
                if rule_id in BUG_RULES and eslint_severity == 1:
                    severity = Severity.MEDIUM

                confidence_score = 0.85 if rule_id in BUG_RULES else 0.65

                finding = FindingSchema(
                    source=AnalysisSource.ESLINT,
                    rule_id=rule_id,
                    file_path=rel_path or file_abs_path,
                    line_number=message.get("line"),
                    column=message.get("column"),
                    severity=severity,
                    confidence=ConfidenceLevel.HIGH if rule_id in BUG_RULES else ConfidenceLevel.MEDIUM,
                    confidence_score=confidence_score,
                    issue=f"[{rule_id}] {message.get('message', '')}",
                    suggestion=self._make_suggestion(rule_id, message.get("message", "")),
                )
                result.findings.append(finding)

        logger.debug("ESLint complete", findings=len(result.findings))
        return result

    def _find_eslint(self, repo_path: str) -> str:
        """Find eslint binary — prefer project-local over global."""
        local = Path(repo_path) / "node_modules" / ".bin" / "eslint"
        if local.exists():
            return str(local)
        return "eslint"

    def _make_suggestion(self, rule_id: str, message: str) -> str:
        suggestions: dict[str, str] = {
            "no-undef": "Define or import the variable before use",
            "no-unused-vars": "Remove the unused variable or export it if intentional",
            "no-unreachable": "Remove unreachable code after return/throw/break/continue",
            "no-eval": "Replace eval() with safer alternatives (JSON.parse, Function constructor with validation)",
            "no-implied-eval": "Avoid passing strings to setTimeout/setInterval — pass function references",
            "no-prototype-builtins": "Use Object.prototype.hasOwnProperty.call(obj, key) instead of obj.hasOwnProperty(key)",
        }
        return suggestions.get(rule_id, f"Fix ESLint rule {rule_id}: {message[:100]}")