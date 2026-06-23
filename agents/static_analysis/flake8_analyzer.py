"""
agents/static_analysis/flake8_analyzer.py

Flake8 analyzer — PEP 8 style and error checking for Python files.

While we deprioritize pure style issues in the LLM review,
flake8 catches real bugs: undefined names (F821), unused imports (F401),
undefined local vars (F823), etc. We include error-category rules
and filter out pure cosmetic ones.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from shared.constants import AnalysisSource, ConfidenceLevel, Severity
from shared.logger import get_logger
from shared.schemas import FileChangeSchema, FindingSchema

logger = get_logger(__name__)

# Flake8 error code prefix → Severity
FLAKE8_SEVERITY_MAP: dict[str, Severity] = {
    "F8": Severity.HIGH,    # F8xx: pyflakes errors (undefined names, etc.)
    "F4": Severity.MEDIUM,  # F4xx: import issues
    "F8": Severity.HIGH,
    "E9": Severity.CRITICAL, # E9xx: syntax errors, can't parse file
    "W6": Severity.MEDIUM,  # W6xx: deprecated features
    "C9": Severity.MEDIUM,  # C9xx: McCabe complexity
}

# Pure style codes to skip
SKIP_CODES: frozenset[str] = frozenset({
    "E1", "E2", "E3", "E4", "E5",  # Indentation, whitespace, blank lines, imports, line length
    "W1", "W2", "W3", "W5",         # Whitespace warnings
})


def _severity_for_code(code: str) -> Severity:
    if code.startswith("E9"):
        return Severity.CRITICAL
    if code.startswith("F8") or code.startswith("F82"):
        return Severity.HIGH
    if code.startswith("F4") or code.startswith("C9") or code.startswith("W6"):
        return Severity.MEDIUM
    return Severity.LOW


def _should_skip(code: str) -> bool:
    """Return True for pure cosmetic codes."""
    for prefix in SKIP_CODES:
        if code.startswith(prefix):
            return True
    return False


class AnalyzerResult:
    def __init__(self):
        self.findings: list[FindingSchema] = []
        self.files_analyzed: list[str] = []


class Flake8Analyzer:
    """Runs flake8 and normalizes findings."""

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
            "python", "-m", "flake8",
            "--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s",
            "--max-line-length=120",
            "--max-complexity=15",
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
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            logger.error("Flake8 timed out")
            return result
        except Exception as exc:
            logger.error("Flake8 subprocess failed", error=str(exc))
            return result

        raw = stdout.decode("utf-8", errors="replace").strip()
        if not raw:
            return result

        for line in raw.splitlines():
            parts = line.split(":", 4)
            if len(parts) < 5:
                continue

            path_str, row_str, col_str, code, text = parts
            code = code.strip()

            if _should_skip(code):
                continue

            try:
                line_num = int(row_str.strip())
                col_num = int(col_str.strip())
            except ValueError:
                continue

            # Convert absolute path to relative
            rel_path = path_str.strip().replace(repo_path.rstrip("/") + "/", "")

            finding = FindingSchema(
                source=AnalysisSource.FLAKE8,
                rule_id=code,
                file_path=rel_path,
                line_number=line_num,
                column=col_num,
                severity=_severity_for_code(code),
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.90,
                issue=f"[{code}] {text.strip()}",
                suggestion=self._make_suggestion(code, text.strip()),
            )
            result.findings.append(finding)

        logger.debug("Flake8 complete", findings=len(result.findings))
        return result

    def _make_suggestion(self, code: str, text: str) -> str:
        suggestions: dict[str, str] = {
            "E901": "Fix the syntax error to allow the file to be parsed",
            "E902": "Fix the tokenization error",
            "F401": "Remove unused import or use __all__ to suppress",
            "F811": "Remove the redefinition of the existing name",
            "F821": "Define the name before use or import it",
            "F841": "Remove the local variable that is assigned but never used",
            "C901": "Refactor the function to reduce cyclomatic complexity",
        }
        return suggestions.get(code, f"Review flake8 rule {code}: {text[:100]}")