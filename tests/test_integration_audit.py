from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agents.merger import MergerAgent
from agents.review import LLMReviewAgent
from agents.security import SecurityAnalysisAgent
from agents.static_analysis import StaticAnalysisAgent
from apps.webhook_api.main import create_app
from github_int.diff_extractor import DiffExtractor
from github_int.github_client import GitHubClient
from shared.config import get_settings
from shared.constants import AnalysisSource, ConfidenceLevel, Severity
from shared.schemas import FileChangeSchema, FindingSchema, LLMReviewResult, PRDiffSchema, SecurityAnalysisResult, StaticAnalysisResult


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_git_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)

    (repo_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True, capture_output=True)

    (repo_path / "app.py").write_text("print('hello world')\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "second"], cwd=repo_path, check=True, capture_output=True)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True).strip()
    base = subprocess.check_output(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo_path, text=True).strip()
    return repo_path, head, base


def test_diff_extraction_extracts_changed_files(tmp_path: Path):
    repo_path, head_sha, base_sha = _write_git_repo(tmp_path)
    extractor = DiffExtractor()

    diff = extractor.extract(
        local_repo_path=str(repo_path),
        repo_full_name="test/repo",
        pr_number=1,
        head_sha=head_sha,
        base_sha=base_sha,
    )

    assert diff.pr_number == 1
    assert len(diff.changed_files) == 1
    assert diff.changed_files[0].file_path == "app.py"
    assert diff.changed_files[0].change_type == "modified"


@pytest.mark.asyncio
async def test_static_analysis_agent_collects_findings():
    agent = StaticAnalysisAgent()
    diff = PRDiffSchema(
        repo_full_name="test/repo",
        pr_number=1,
        head_sha="a",
        base_sha="b",
        local_repo_path=".",
        changed_files=[FileChangeSchema(file_path="app.py", change_type="modified", patch="diff")],
    )

    class DummyResult:
        def __init__(self):
            self.findings = [FindingSchema(source=AnalysisSource.PYLINT, file_path="app.py", line_number=1, severity=Severity.HIGH, confidence=ConfidenceLevel.HIGH, confidence_score=0.9, issue="bad", suggestion="fix it")]
            self.files_analyzed = ["app.py"]

    async def fake_analyze(repo_path, files):
        return DummyResult()

    agent._pylint.analyze = fake_analyze  # type: ignore[assignment]

    result = await agent.analyze(diff)

    assert result.findings
    assert result.findings[0].file_path == "app.py"


@pytest.mark.asyncio
async def test_security_analysis_agent_collects_findings():
    agent = SecurityAnalysisAgent()
    diff = PRDiffSchema(
        repo_full_name="test/repo",
        pr_number=1,
        head_sha="a",
        base_sha="b",
        local_repo_path=".",
        changed_files=[FileChangeSchema(file_path="app.py", change_type="modified", patch="diff")],
    )

    async def fake_run_bandit(repo_path, files):
        return ([FindingSchema(source=AnalysisSource.BANDIT, file_path="app.py", line_number=1, severity=Severity.HIGH, confidence=ConfidenceLevel.HIGH, confidence_score=0.9, issue="unsafe", suggestion="fix")], [], ["app.py"])

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(agent, "_run_bandit", fake_run_bandit)
    try:
        result = await agent.analyze(diff)
    finally:
        monkeypatch.undo()

    assert result.findings
    assert result.findings[0].source == AnalysisSource.BANDIT


@pytest.mark.asyncio
async def test_llm_review_agent_parses_structured_output():
    agent = LLMReviewAgent()
    diff = PRDiffSchema(
        repo_full_name="test/repo",
        pr_number=1,
        head_sha="a",
        base_sha="b",
        local_repo_path=".",
        changed_files=[FileChangeSchema(file_path="app.py", change_type="modified", patch="diff --git a/app.py b/app.py\n+print('x')")],
    )

    class FakeResponse:
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)

        class Choice:
            class Message:
                content = json.dumps([
                    {
                        "file_path": "app.py",
                        "line_number": 1,
                        "severity": "HIGH",
                        "confidence": "HIGH",
                        "issue": "Potential bug",
                        "suggestion": "Fix it",
                    }
                ])

            message = Message()

        choices = [Choice()]

    class FakeCompletions:
        async def create(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    agent._client = FakeClient()  # type: ignore[assignment]

    result = await agent.review(diff)

    assert result.findings
    assert result.findings[0].issue == "Potential bug"


def test_merge_agent_deduplicates_and_ranks():
    agent = MergerAgent()
    static = StaticAnalysisResult(findings=[FindingSchema(source=AnalysisSource.PYLINT, file_path="app.py", line_number=1, severity=Severity.MEDIUM, confidence=ConfidenceLevel.MEDIUM, confidence_score=0.6, issue="same", suggestion="fix")])
    security = SecurityAnalysisResult(findings=[FindingSchema(source=AnalysisSource.SEMGREP, file_path="app.py", line_number=1, severity=Severity.HIGH, confidence=ConfidenceLevel.HIGH, confidence_score=0.9, issue="same", suggestion="fix")])
    llm = LLMReviewResult(findings=[FindingSchema(source=AnalysisSource.LLM_REVIEW, file_path="app.py", line_number=1, severity=Severity.HIGH, confidence=ConfidenceLevel.HIGH, confidence_score=0.95, issue="same", suggestion="fix")])

    merged = agent.merge(static, security, llm)

    assert len(merged.findings) == 1
    assert merged.findings[0].source == AnalysisSource.LLM_REVIEW
    assert merged.high_count == 1


def test_github_client_posts_review_comments():
    client = GitHubClient()

    class FakeReview:
        def __init__(self):
            self.calls = []

        def create_review(self, **kwargs):
            self.calls.append(kwargs)

        def create_issue_comment(self, body):
            self.calls.append({"body": body})

    class FakeRepo:
        def __init__(self):
            self.review = FakeReview()

        def get_pull(self, pr_number):
            return self.review

        def get_commit(self, head_sha):
            return {"sha": head_sha}

    fake_repo = FakeRepo()

    with patch("github_int.github_client._get_repo", return_value=fake_repo):
        async def fake_run_sync(self, func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(GitHubClient, "_run_sync", new=fake_run_sync):
            asyncio.run(client.post_review("test/repo", 1, SimpleNamespace(findings=[FindingSchema(source=AnalysisSource.PYLINT, file_path="app.py", line_number=1, severity=Severity.HIGH, confidence=ConfidenceLevel.HIGH, confidence_score=0.9, issue="bad", suggestion="fix")], markdown_report="report"), "abc"))

    assert fake_repo.review.calls


def _sign_payload(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_webhook_route_enqueues_review_job():
    env = {
        "GITHUB_WEBHOOK_SECRET": "test_secret",
        "GITHUB_TOKEN": "ghp_test",
        "SECRET_KEY": "test_secret",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "REDIS_URL": "redis://localhost:6379/0",
        "OPENAI_API_KEY": "test_key",
    }
    with patch.dict(os.environ, env, clear=False):
        get_settings.cache_clear()
        app = create_app()
        with TestClient(app) as client:
            body = json.dumps({
                "action": "opened",
                "pull_request": {
                    "number": 42,
                    "title": "Test PR",
                    "body": "desc",
                    "html_url": "https://example.com/pr/42",
                    "state": "open",
                    "user": {"login": "alice"},
                    "head": {"sha": "abc", "ref": "feature"},
                    "base": {"sha": "def", "ref": "main"},
                },
                "repository": {
                    "id": 1,
                    "name": "repo",
                    "full_name": "test/repo",
                    "clone_url": "https://example.com/repo.git",
                    "default_branch": "main",
                    "language": "Python",
                },
                "installation": {"id": 10},
            }).encode("utf-8")
            sig = _sign_payload(body, "test_secret")
            with patch("apps.webhook_api.routes.webhook.TaskQueue.enqueue", new_callable=AsyncMock) as mock_enqueue:
                mock_enqueue.return_value = "job-123"
                response = client.post(
                    "/webhooks/github",
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Hub-Signature-256": sig,
                        "X-GitHub-Event": "pull_request",
                        "X-GitHub-Delivery": "delivery-1",
                    },
                )
        assert response.status_code in {200, 202}
        assert response.json()["job_id"] == "job-123"
