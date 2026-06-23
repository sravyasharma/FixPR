"""
tests/test_merger.py

Unit tests for the MergerAgent.

Tests:
  - Deduplication of identical findings from different sources
  - Severity ranking order
  - Confidence tie-breaking
  - Markdown report generation
  - Severity counts
"""

from __future__ import annotations

import pytest

from agents.merger import MergerAgent
from shared.constants import AnalysisSource, Severity
from shared.schemas import (
    LLMReviewResult,
    MergedReviewResult,
    SecurityAnalysisResult,
    StaticAnalysisResult,
)
from tests.conftest import make_finding


class TestMergerDeduplication:
    """Deduplication logic tests."""

    def test_same_finding_from_two_sources_is_deduplicated(self):
        """Two findings at same file+line with same issue → keep one."""
        merger = MergerAgent()

        finding_a = make_finding(
            source=AnalysisSource.PYLINT,
            file_path="app.py",
            line_number=10,
            issue="Hardcoded secret key detected",
            confidence_score=0.80,
        )
        finding_b = make_finding(
            source=AnalysisSource.BANDIT,
            file_path="app.py",
            line_number=10,
            issue="Hardcoded secret key detected",  # Same issue text
            confidence_score=0.90,  # Higher confidence
        )

        result = merger.merge(
            static_result=StaticAnalysisResult(findings=[finding_a]),
            security_result=SecurityAnalysisResult(findings=[finding_b]),
            llm_result=LLMReviewResult(),
        )

        assert result.deduplicated_count == 1
        assert len(result.findings) == 1
        # Should keep BANDIT finding (higher confidence)
        assert result.findings[0].source == AnalysisSource.BANDIT
        assert result.findings[0].confidence_score == 0.90

    def test_different_lines_not_deduplicated(self):
        """Findings on different lines are distinct."""
        merger = MergerAgent()

        finding_a = make_finding(file_path="app.py", line_number=10, issue="Unused variable x")
        finding_b = make_finding(file_path="app.py", line_number=20, issue="Unused variable x")

        result = merger.merge(
            static_result=StaticAnalysisResult(findings=[finding_a, finding_b]),
            security_result=SecurityAnalysisResult(),
            llm_result=LLMReviewResult(),
        )

        assert len(result.findings) == 2
        assert result.deduplicated_count == 0

    def test_different_files_not_deduplicated(self):
        """Same issue in different files = distinct findings."""
        merger = MergerAgent()

        finding_a = make_finding(file_path="a.py", line_number=5, issue="sql injection risk")
        finding_b = make_finding(file_path="b.py", line_number=5, issue="sql injection risk")

        result = merger.merge(
            static_result=StaticAnalysisResult(findings=[finding_a, finding_b]),
            security_result=SecurityAnalysisResult(),
            llm_result=LLMReviewResult(),
        )

        assert len(result.findings) == 2

    def test_rule_prefix_stripped_for_dedup(self):
        """[B101] and [E501] prefixes are stripped before comparison."""
        merger = MergerAgent()

        # Same logical issue, different tool prefixes
        finding_a = make_finding(
            source=AnalysisSource.PYLINT,
            issue="[E0001] Syntax error in file",
            line_number=3,
        )
        finding_b = make_finding(
            source=AnalysisSource.FLAKE8,
            issue="[E999] Syntax error in file",  # Different prefix, same message
            line_number=3,
        )

        result = merger.merge(
            static_result=StaticAnalysisResult(findings=[finding_a, finding_b]),
            security_result=SecurityAnalysisResult(),
            llm_result=LLMReviewResult(),
        )

        assert result.deduplicated_count == 1


class TestMergerRanking:
    """Severity ranking tests."""

    def test_critical_findings_come_first(self):
        """CRITICAL > HIGH > MEDIUM > LOW > INFO ordering."""
        merger = MergerAgent()

        findings = [
            make_finding(severity=Severity.INFO, issue="Info level note", line_number=1),
            make_finding(severity=Severity.CRITICAL, issue="Critical bug", line_number=2),
            make_finding(severity=Severity.LOW, issue="Low severity issue", line_number=3),
            make_finding(severity=Severity.HIGH, issue="High severity bug", line_number=4),
            make_finding(severity=Severity.MEDIUM, issue="Medium issue", line_number=5),
        ]

        result = merger.merge(
            static_result=StaticAnalysisResult(findings=findings),
            security_result=SecurityAnalysisResult(),
            llm_result=LLMReviewResult(),
        )

        severities = [f.severity for f in result.findings]
        expected_order = [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
            Severity.INFO,
        ]
        assert severities == expected_order

    def test_same_severity_ordered_by_confidence(self):
        """Within same severity, higher confidence appears first."""
        merger = MergerAgent()

        low_confidence = make_finding(
            severity=Severity.HIGH,
            issue="Issue A",
            confidence_score=0.50,
            line_number=1,
        )
        high_confidence = make_finding(
            severity=Severity.HIGH,
            issue="Issue B",
            confidence_score=0.95,
            line_number=2,
        )

        result = merger.merge(
            static_result=StaticAnalysisResult(findings=[low_confidence, high_confidence]),
            security_result=SecurityAnalysisResult(),
            llm_result=LLMReviewResult(),
        )

        assert result.findings[0].confidence_score == 0.95
        assert result.findings[1].confidence_score == 0.50


class TestMergerCounts:
    """Severity count tests."""

    def test_counts_are_accurate(self, critical_finding, high_finding, medium_finding):
        merger = MergerAgent()

        result = merger.merge(
            static_result=StaticAnalysisResult(findings=[medium_finding]),
            security_result=SecurityAnalysisResult(findings=[critical_finding, high_finding]),
            llm_result=LLMReviewResult(),
        )

        assert result.critical_count == 1
        assert result.high_count == 1
        assert result.medium_count == 1
        assert result.low_count == 0
        assert result.info_count == 0
        assert len(result.findings) == 3

    def test_empty_inputs_produce_empty_result(self):
        merger = MergerAgent()

        result = merger.merge(
            static_result=StaticAnalysisResult(),
            security_result=SecurityAnalysisResult(),
            llm_result=LLMReviewResult(),
        )

        assert result.total_before_dedup == 0
        assert len(result.findings) == 0
        assert result.critical_count == 0


class TestMergerMarkdown:
    """Markdown report generation tests."""

    def test_report_contains_severity_sections(self, critical_finding, high_finding):
        merger = MergerAgent()

        result = merger.merge(
            static_result=StaticAnalysisResult(),
            security_result=SecurityAnalysisResult(findings=[critical_finding, high_finding]),
            llm_result=LLMReviewResult(),
        )

        assert "# 🤖 AI Code Review Report" in result.markdown_report
        assert "CRITICAL" in result.markdown_report
        assert "HIGH" in result.markdown_report

    def test_no_findings_shows_pass_message(self):
        merger = MergerAgent()

        result = merger.merge(
            static_result=StaticAnalysisResult(),
            security_result=SecurityAnalysisResult(),
            llm_result=LLMReviewResult(),
        )

        assert "No Issues Found" in result.markdown_report