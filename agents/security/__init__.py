"""
agents/security/__init__.py

Security Analysis Agent — orchestrates Bandit and Semgrep.

Both tools produce SARIF output which is parsed into normalized FindingSchema.
OWASP Top 10 rule packs are loaded for Semgrep.

Architecture role:
  - Runs concurrently with StaticAnalysisAgent and LLMReviewAgent
  - Outputs SecurityAnalysisResult with normalized findings + raw SARIF
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from shared.config import get_settings
from shared.constants import AnalysisSource, ConfidenceLevel, Severity
from shared.logger import get_logger
from shared.schemas import FileChangeSchema, FindingSchema, PRDiffSchema, SecurityAnalysisResult

logger = get_logger(__name__)


# Bandit severity/confidence → our Severity
BANDIT_SEVERITY_MAP: dict[str, Severity] = {
    "HIGH": Severity.HIGH,
    "MEDIUM": Severity.MEDIUM,
    "LOW": Severity.LOW,
}

BANDIT_CONFIDENCE_MAP: dict[str, tuple[ConfidenceLevel, float]] = {
    "HIGH": (ConfidenceLevel.HIGH, 0.90),
    "MEDIUM": (ConfidenceLevel.MEDIUM, 0.65),
    "LOW": (ConfidenceLevel.LOW, 0.35),
}

# Semgrep SARIF severity → our Severity
SARIF_SEVERITY_MAP: dict[str, Severity] = {
    "error": Severity.HIGH,
    "warning": Severity.MEDIUM,
    "note": Severity.LOW,
    "none": Severity.INFO,
}


class SecurityAnalysisAgent:
    """
    Runs Bandit and Semgrep over changed files.

    Bandit: Python-specific security linter
    Semgrep: Multi-language pattern matching with OWASP rules
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def analyze(self, diff: PRDiffSchema) -> SecurityAnalysisResult:
        start = time.monotonic()

        tasks = []
        tool_names = []

        # Filter out deleted files
        active_files = [f for f in diff.changed_files if f.change_type != "deleted"]

        python_files = [
            f for f in active_files
            if Path(f.file_path).suffix.lower() in {".py", ".pyi"}
        ]

        if self._settings.bandit_enabled and python_files:
            tasks.append(self._run_bandit(diff.local_repo_path, python_files))
            tool_names.append("bandit")

        if self._settings.semgrep_enabled and active_files:
            tasks.append(self._run_semgrep(diff.local_repo_path, active_files))
            tool_names.append("semgrep")

        if not tasks:
            return SecurityAnalysisResult(duration_seconds=time.monotonic() - start)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_findings: list[FindingSchema] = []
        all_sarif: list[dict[str, Any]] = []
        errors: list[str] = []
        files_analyzed: set[str] = set()
        analyzers_run: list[str] = []

        for tool_name, result in zip(tool_names, results):
            if isinstance(result, Exception):
                errors.append(f"{tool_name}: {str(result)}")
                logger.error("Security tool failed", tool=tool_name, error=str(result))
            else:
                findings, sarif_data, tool_files = result
                all_findings.extend(findings)
                all_sarif.extend(sarif_data)
                files_analyzed.update(tool_files)
                analyzers_run.append(tool_name)
                logger.info(
                    "Security tool completed",
                    tool=tool_name,
                    findings=len(findings),
                )

        duration = time.monotonic() - start
        logger.info(
            "Security analysis complete",
            total_findings=len(all_findings),
            analyzers_run=analyzers_run,
            duration=f"{duration:.2f}s",
        )

        return SecurityAnalysisResult(
            findings=all_findings,
            files_analyzed=list(files_analyzed),
            analyzers_run=analyzers_run,
            sarif_reports=all_sarif,
            errors=errors,
            duration_seconds=duration,
        )

    # ---------------------------------------------------------------- #
    # Bandit
    # ---------------------------------------------------------------- #

    async def _run_bandit(
        self, repo_path: str, files: list[FileChangeSchema]
    ) -> tuple[list[FindingSchema], list[dict], list[str]]:
        """Run bandit with SARIF output and parse results."""
        file_paths = [
            str(Path(repo_path) / f.file_path)
            for f in files
            if (Path(repo_path) / f.file_path).exists()
        ]

        import tempfile
        sarif_file = tempfile.NamedTemporaryFile(suffix=".sarif.json", delete=False)
        sarif_path = sarif_file.name
        sarif_file.close()

        cmd = [
            "python", "-m", "bandit",
            "--format", "sarif",
            "--output", sarif_path,
            "--recursive",
            "--severity-level", "low",
            "--confidence-level", "low",
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
            await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            logger.error("Bandit timed out")
            return [], [], []
        except Exception as exc:
            logger.error("Bandit subprocess failed", error=str(exc))
            return [], [], []

        # Parse SARIF output
        try:
            with open(sarif_path, "r") as f:
                sarif_data = json.load(f)
        except Exception as exc:
            logger.warning("Could not read Bandit SARIF output", error=str(exc))
            return [], [], [f.file_path for f in files]
        finally:
            Path(sarif_path).unlink(missing_ok=True)

        findings = self._parse_sarif(sarif_data, repo_path, AnalysisSource.BANDIT)
        return findings, [sarif_data], [f.file_path for f in files]

    # ---------------------------------------------------------------- #
    # Semgrep
    # ---------------------------------------------------------------- #

    async def _run_semgrep(
        self, repo_path: str, files: list[FileChangeSchema]
    ) -> tuple[list[FindingSchema], list[dict], list[str]]:
        """Run Semgrep with OWASP rules and parse SARIF output."""
        rules = self._settings.semgrep_rules
        file_paths = [
            str(Path(repo_path) / f.file_path)
            for f in files
            if (Path(repo_path) / f.file_path).exists()
        ]

        if not file_paths:
            return [], [], []

        cmd = [
            "semgrep",
            "--config", rules,
            "--sarif",
            "--quiet",
            "--no-git-ignore",
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
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
        except asyncio.TimeoutError:
            logger.error("Semgrep timed out")
            return [], [], []
        except FileNotFoundError:
            logger.warning("Semgrep not installed — skipping")
            return [], [], []
        except Exception as exc:
            logger.error("Semgrep subprocess failed", error=str(exc))
            return [], [], []

        raw = stdout.decode("utf-8", errors="replace").strip()
        if not raw:
            return [], [], [f.file_path for f in files]

        try:
            sarif_data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Semgrep SARIF", raw=raw[:500])
            return [], [], [f.file_path for f in files]

        findings = self._parse_sarif(sarif_data, repo_path, AnalysisSource.SEMGREP)
        return findings, [sarif_data], [f.file_path for f in files]

    # ---------------------------------------------------------------- #
    # SARIF Parser
    # ---------------------------------------------------------------- #

    def _parse_sarif(
        self,
        sarif: dict[str, Any],
        repo_path: str,
        source: AnalysisSource,
    ) -> list[FindingSchema]:
        """
        Parse a SARIF 2.1.0 document into normalized FindingSchema objects.

        SARIF structure:
          sarif.runs[].results[].{
            ruleId, message, locations[].physicalLocation, level
          }
          sarif.runs[].tool.driver.rules[ruleId].{
            shortDescription, fullDescription, helpUri
          }
        """
        findings: list[FindingSchema] = []

        for run in sarif.get("runs", []):
            # Build rule lookup from tool.driver.rules
            rules_by_id: dict[str, dict[str, Any]] = {}
            driver = run.get("tool", {}).get("driver", {})
            for rule in driver.get("rules", []):
                rules_by_id[rule.get("id", "")] = rule

            for result in run.get("results", []):
                rule_id: str = result.get("ruleId", "")
                level: str = result.get("level", "warning")
                message: str = (
                    result.get("message", {}).get("text", "")
                    or result.get("message", {}).get("markdown", "")
                )

                # Extract location
                locations = result.get("locations", [])
                file_path = ""
                line_number = None
                end_line = None
                if locations:
                    loc = locations[0]
                    phys = loc.get("physicalLocation", {})
                    artifact = phys.get("artifactLocation", {})
                    uri = artifact.get("uri", "")
                    # Strip repo_path prefix and file:// scheme
                    file_path = uri.replace(f"file://{repo_path}/", "").replace(f"{repo_path}/", "")

                    region = phys.get("region", {})
                    line_number = region.get("startLine")
                    end_line = region.get("endLine")

                # Look up rule metadata for better suggestion
                rule_meta = rules_by_id.get(rule_id, {})
                help_uri = rule_meta.get("helpUri")
                short_desc = (
                    rule_meta.get("shortDescription", {}).get("text")
                    or rule_meta.get("fullDescription", {}).get("text")
                    or ""
                )

                severity = SARIF_SEVERITY_MAP.get(level, Severity.MEDIUM)

                finding = FindingSchema(
                    source=source,
                    rule_id=rule_id or None,
                    file_path=file_path or "unknown",
                    line_number=line_number,
                    end_line_number=end_line,
                    severity=severity,
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.85,
                    issue=f"[{rule_id}] {message}",
                    suggestion=short_desc or f"Review {source} rule {rule_id}",
                    sarif_rule_url=help_uri,
                )
                findings.append(finding)

        return findings