"""
agents/static_analysis/__init__.py

Static Analysis Agent — orchestrates pylint, flake8, mypy, and eslint.

Each analyzer is isolated in its own module and returns normalized FindingSchema objects.
The agent runs all applicable analyzers concurrently via asyncio.gather().

Architecture role:
  - Called concurrently with SecurityAgent and LLMReviewAgent by the Orchestrator
  - Receives PRDiffSchema (repo path + changed files)
  - Returns StaticAnalysisResult (list of normalized FindingSchema)
  - Only analyzes files that actually changed in the PR
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from shared.config import get_settings
from shared.constants import PYTHON_EXTENSIONS, JS_TS_EXTENSIONS
from shared.logger import get_logger
from shared.schemas import (
    FileChangeSchema,
    PRDiffSchema,
    StaticAnalysisResult,
)
from agents.static_analysis.pylint_analyzer import PylintAnalyzer
from agents.static_analysis.flake8_analyzer import Flake8Analyzer
from agents.static_analysis.mypy_analyzer import MypyAnalyzer
from agents.static_analysis.eslint_analyzer import ESLintAnalyzer

logger = get_logger(__name__)


class StaticAnalysisAgent:
    """
    Runs static analysis tools over PR-changed files.

    Analyzer selection is file-type aware:
      - Python files → pylint + flake8 + mypy
      - JS/TS files  → eslint

    All applicable analyzers run concurrently.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._pylint = PylintAnalyzer()
        self._flake8 = Flake8Analyzer()
        self._mypy = MypyAnalyzer()
        self._eslint = ESLintAnalyzer()

    async def analyze(self, diff: PRDiffSchema) -> StaticAnalysisResult:
        """
        Run all enabled static analyzers over changed files.

        Returns aggregated StaticAnalysisResult.
        """
        start = time.monotonic()

        python_files = self._filter_files(diff, PYTHON_EXTENSIONS)
        js_ts_files = self._filter_files(diff, JS_TS_EXTENSIONS)

        tasks = []
        analyzer_names = []

        if self._settings.pylint_enabled and python_files:
            tasks.append(self._pylint.analyze(diff.local_repo_path, python_files))
            analyzer_names.append("pylint")

        if self._settings.flake8_enabled and python_files:
            tasks.append(self._flake8.analyze(diff.local_repo_path, python_files))
            analyzer_names.append("flake8")

        if self._settings.mypy_enabled and python_files:
            tasks.append(self._mypy.analyze(diff.local_repo_path, python_files))
            analyzer_names.append("mypy")

        if self._settings.eslint_enabled and js_ts_files:
            tasks.append(self._eslint.analyze(diff.local_repo_path, js_ts_files))
            analyzer_names.append("eslint")

        if not tasks:
            logger.info(
                "No applicable static analyzers for changed files",
                changed_files=diff.changed_file_paths,
            )
            return StaticAnalysisResult(duration_seconds=time.monotonic() - start)

        # Run all analyzers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_findings = []
        errors = []
        files_analyzed: set[str] = set()
        analyzers_run = []

        for name, result in zip(analyzer_names, results):
            if isinstance(result, Exception):
                error_msg = f"{name}: {str(result)}"
                errors.append(error_msg)
                logger.error("Static analyzer failed", analyzer=name, error=str(result))
            else:
                all_findings.extend(result.findings)
                files_analyzed.update(result.files_analyzed)
                analyzers_run.append(name)
                logger.info(
                    "Analyzer completed",
                    analyzer=name,
                    findings=len(result.findings),
                )

        duration = time.monotonic() - start
        logger.info(
            "Static analysis complete",
            total_findings=len(all_findings),
            analyzers_run=analyzers_run,
            duration=f"{duration:.2f}s",
        )

        return StaticAnalysisResult(
            findings=all_findings,
            files_analyzed=list(files_analyzed),
            analyzers_run=analyzers_run,
            errors=errors,
            duration_seconds=duration,
        )

    def _filter_files(
        self, diff: PRDiffSchema, extensions: frozenset[str]
    ) -> list[FileChangeSchema]:
        """Return only changed files matching the given extensions."""
        return [
            f
            for f in diff.changed_files
            if Path(f.file_path).suffix.lower() in extensions
            and f.change_type != "deleted"
        ]