"""
agents/merger/__init__.py

Merger Agent — consolidates outputs from all three analysis agents.

Responsibilities:
  1. Deduplication   — collapse same logical issue from multiple sources
  2. Severity Ranking — sort by SeverityScore (CRITICAL=5 down to INFO=1)
  3. Confidence Scoring — when deduplicating, keep highest-confidence finding
  4. Markdown Report — generate PR-ready review comment with full summary

Dedup key: (file_path, line_number, normalized_issue_text)
When two findings share a key, keep the one with higher confidence_score.
If scores are equal, prefer: LLM_REVIEW > SECURITY > STATIC sources.

Architecture role:
  - Called by the Orchestrator AFTER all three agents complete
  - Receives three result objects, returns MergedReviewResult
  - MergedReviewResult is consumed by GitHubClient and storage layer
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict

from shared.constants import (
    AnalysisSource,
    Severity,
    SeverityScore,
)
from shared.logger import get_logger
from shared.schemas import (
    FindingSchema,
    LLMReviewResult,
    MergedReviewResult,
    SecurityAnalysisResult,
    StaticAnalysisResult,
)

logger = get_logger(__name__)

# Source priority for tie-breaking during deduplication
SOURCE_PRIORITY: dict[str, int] = {
    AnalysisSource.LLM_REVIEW: 3,
    AnalysisSource.BANDIT: 2,
    AnalysisSource.SEMGREP: 2,
    AnalysisSource.MYPY: 2,
    AnalysisSource.PYLINT: 1,
    AnalysisSource.FLAKE8: 1,
    AnalysisSource.ESLINT: 1,
}

# Severity emoji for markdown report
SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH:     "🟠",
    Severity.MEDIUM:   "🟡",
    Severity.LOW:      "🔵",
    Severity.INFO:     "⚪",
}


class MergerAgent:
    """
    Merges findings from Static, Security, and LLM Review agents.

    Usage:
        merger = MergerAgent()
        result = merger.merge(static_result, security_result, llm_result)
    """

    def merge(
        self,
        static_result: StaticAnalysisResult,
        security_result: SecurityAnalysisResult,
        llm_result: LLMReviewResult,
    ) -> MergedReviewResult:
        start = time.monotonic()

        # 1. Collect all findings from all sources
        all_findings: list[FindingSchema] = (
            static_result.findings
            + security_result.findings
            + llm_result.findings
        )
        total_before_dedup = len(all_findings)

        logger.info(
            "Starting merge",
            static_count=len(static_result.findings),
            security_count=len(security_result.findings),
            llm_count=len(llm_result.findings),
            total=total_before_dedup,
        )

        # 2. Assign dedup keys to all findings
        keyed = [self._assign_dedup_key(f) for f in all_findings]

        # 3. Deduplicate — keep highest-confidence finding per key
        deduped = self._deduplicate(keyed)

        # 4. Rank by severity then confidence
        ranked = self._rank(deduped)

        # 5. Count by severity
        counts = self._count_by_severity(ranked)

        # 6. Generate markdown report
        report = self._generate_markdown_report(
            ranked,
            static_result,
            security_result,
            llm_result,
            counts,
        )

        duration = time.monotonic() - start
        logger.info(
            "Merge complete",
            before_dedup=total_before_dedup,
            after_dedup=len(ranked),
            deduplicated=total_before_dedup - len(ranked),
            duration=f"{duration:.2f}s",
        )

        return MergedReviewResult(
            findings=ranked,
            deduplicated_count=total_before_dedup - len(ranked),
            total_before_dedup=total_before_dedup,
            critical_count=counts[Severity.CRITICAL],
            high_count=counts[Severity.HIGH],
            medium_count=counts[Severity.MEDIUM],
            low_count=counts[Severity.LOW],
            info_count=counts[Severity.INFO],
            markdown_report=report,
            duration_seconds=duration,
        )

    # ---------------------------------------------------------------- #
    # Deduplication
    # ---------------------------------------------------------------- #

    def _assign_dedup_key(self, finding: FindingSchema) -> FindingSchema:
        """
        Compute a stable dedup key for a finding.

        Key: SHA-256 of (file_path | line_number | normalized_issue)

        Normalization: lowercase, strip whitespace, remove rule-id prefixes
        like [E501] or [B101] so the same underlying issue from two tools
        maps to the same key.
        """
        issue_normalized = self._normalize_issue(finding.issue)
        line = str(finding.line_number or "")
        raw = f"{finding.file_path}|{line}|{issue_normalized}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return finding.model_copy(update={"dedup_key": key})

    def _normalize_issue(self, issue: str) -> str:
        """Strip tool prefixes and normalize text for comparison."""
        import re
        # Remove [RULE_ID] prefixes: [E501], [B101], [mypy:attr-defined]
        normalized = re.sub(r"^\[[^\]]+\]\s*", "", issue)
        normalized = normalized.lower().strip()
        # Collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _deduplicate(self, findings: list[FindingSchema]) -> list[FindingSchema]:
        """
        For each dedup key, keep only the best finding.

        'Best' = highest confidence_score.
        Tie-break: higher SOURCE_PRIORITY wins.
        """
        buckets: dict[str, list[FindingSchema]] = defaultdict(list)
        for f in findings:
            key = f.dedup_key or f.id  # fallback to ID if no key
            buckets[key].append(f)

        result: list[FindingSchema] = []
        for key, group in buckets.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                best = max(
                    group,
                    key=lambda f: (
                        f.confidence_score,
                        SOURCE_PRIORITY.get(str(f.source), 0),
                    ),
                )
                result.append(best)

        return result

    # ---------------------------------------------------------------- #
    # Ranking
    # ---------------------------------------------------------------- #

    def _rank(self, findings: list[FindingSchema]) -> list[FindingSchema]:
        """
        Sort findings by:
          1. Severity score (CRITICAL=5 first)
          2. Confidence score (descending)
          3. File path (alphabetical, for stable output)
        """
        return sorted(
            findings,
            key=lambda f: (
                -SeverityScore.from_severity(f.severity),
                -f.confidence_score,
                f.file_path,
                f.line_number or 0,
            ),
        )

    # ---------------------------------------------------------------- #
    # Counting
    # ---------------------------------------------------------------- #

    def _count_by_severity(
        self, findings: list[FindingSchema]
    ) -> dict[str, int]:
        counts: dict[str, int] = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 0,
            Severity.MEDIUM: 0,
            Severity.LOW: 0,
            Severity.INFO: 0,
        }
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    # ---------------------------------------------------------------- #
    # Markdown Report
    # ---------------------------------------------------------------- #

    def _generate_markdown_report(
        self,
        findings: list[FindingSchema],
        static_result: StaticAnalysisResult,
        security_result: SecurityAnalysisResult,
        llm_result: LLMReviewResult,
        counts: dict[str, int],
    ) -> str:
        total = len(findings)
        lines: list[str] = []

        # ---- Header ----
        lines += [
            "# 🤖 AI Code Review Report",
            "",
            "Automated analysis by the **AI Code Review Platform** using static analysis, "
            "security scanning, and GPT-4o review.",
            "",
        ]

        # ---- Summary badges ----
        badge_parts = []
        for sev, emoji in SEVERITY_EMOJI.items():
            count = counts.get(sev, 0)
            if count > 0:
                badge_parts.append(f"{emoji} **{count} {sev}**")

        if badge_parts:
            lines.append("## Summary\n")
            lines.append(" &nbsp;|&nbsp; ".join(badge_parts))
        else:
            lines.append("## ✅ No Issues Found\n")
            lines.append("All checks passed. Great work!")

        lines += ["", "---", ""]

        # ---- Tools run ----
        lines.append("## Analysis Tools\n")
        tools_table = [
            "| Tool | Type | Findings |",
            "|------|------|----------|",
        ]
        for tool in static_result.analyzers_run:
            tool_findings = sum(1 for f in findings if str(f.source) == tool)
            tools_table.append(f"| `{tool}` | Static Analysis | {tool_findings} |")
        for tool in security_result.analyzers_run:
            tool_findings = sum(1 for f in findings if str(f.source) == tool)
            tools_table.append(f"| `{tool}` | Security Analysis | {tool_findings} |")
        if llm_result.files_reviewed:
            llm_count = sum(1 for f in findings if f.source == "llm_review")
            tools_table.append(
                f"| `GPT-4o ({llm_result.model_used})` | LLM Review | {llm_count} |"
            )
        lines += tools_table
        lines += ["", "---", ""]

        if not findings:
            return "\n".join(lines)

        # ---- Findings by severity ----
        lines.append("## Findings\n")

        for severity in [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]:
            sev_findings = [f for f in findings if f.severity == severity]
            if not sev_findings:
                continue

            emoji = SEVERITY_EMOJI[severity]
            lines.append(f"### {emoji} {severity} ({len(sev_findings)})\n")

            for i, finding in enumerate(sev_findings, 1):
                location = f"`{finding.file_path}`"
                if finding.line_number:
                    location += f" line {finding.line_number}"

                lines += [
                    f"<details>",
                    f"<summary><b>[{i}] {finding.issue[:100]}</b> — {location}</summary>",
                    "",
                    f"**Source:** `{finding.source}`"
                    + (f" rule `{finding.rule_id}`" if finding.rule_id else ""),
                    f"  ",
                    f"**Location:** {location}",
                    f"  ",
                    f"**Confidence:** {finding.confidence} ({finding.confidence_score:.0%})",
                    f"  ",
                    f"**Issue:** {finding.issue}",
                    f"  ",
                    f"**Suggestion:** {finding.suggestion}",
                ]

                if finding.code_snippet:
                    lines += ["  ", "```", finding.code_snippet, "```"]

                if finding.sarif_rule_url:
                    lines.append(f"  ")
                    lines.append(f"🔗 [Rule documentation]({finding.sarif_rule_url})")

                lines += ["", "</details>", ""]

        # ---- Cost info ----
        if llm_result.cost_usd > 0:
            lines += [
                "---",
                "",
                "<details>",
                "<summary>💰 LLM Cost</summary>",
                "",
                f"| Model | Prompt Tokens | Completion Tokens | Cost |",
                f"|-------|--------------|-------------------|------|",
                f"| `{llm_result.model_used}` | {llm_result.prompt_tokens:,} "
                f"| {llm_result.completion_tokens:,} | ${llm_result.cost_usd:.4f} |",
                "",
                "</details>",
                "",
            ]

        # ---- Footer ----
        lines += [
            "---",
            "",
            f"*Generated by AI Code Review Platform · "
            f"{total} findings ({counts.get(Severity.CRITICAL, 0) + counts.get(Severity.HIGH, 0)} "
            f"actionable)*",
        ]

        return "\n".join(lines)