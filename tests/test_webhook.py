"""
tests/test_webhook.py

Integration tests for the webhook API routes.
Uses FastAPI TestClient with mocked queue.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from apps.webhook_api.main import create_app

# A known test secret used for signature generation in tests
TEST_WEBHOOK_SECRET = "test_webhook_secret_key"
TEST_GITHUB_TOKEN = "ghp_test_token"


@pytest.fixture(scope="module")
def test_app():
    """Create a test FastAPI app with overridden settings."""
    with patch("shared.config.Settings.github_webhook_secret", TEST_WEBHOOK_SECRET), \
         patch("shared.config.Settings.github_token", TEST_GITHUB_TOKEN), \
         patch("shared.config.Settings.secret_key", "test_secret"), \
         patch("shared.config.Settings.database_url", "sqlite+aiosqlite:///:memory:"), \
         patch("shared.config.Settings.redis_url", "redis://localhost:6379/0"), \
         patch("shared.config.Settings.openai_api_key", "test_key"):
        app = create_app()
        yield app


def _sign_payload(body: bytes, secret: str) -> str:
    """Generate a GitHub-style HMAC-SHA256 signature."""
    sig = hmac.new(
        key=secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"sha256={sig}"


def _make_pr_payload(action: str = "opened") -> dict:
    """Construct a minimal GitHub pull_request webhook payload."""
    return {
        "action": action,
        "pull_request": {
            "number": 42,
            "title": "Test PR",
            "body": "Test description",
            "html_url": "https://github.com/testorg/testrepo/pull/42",
            "state": "open",
            "user": {"login": "testuser"},
            "head": {
                "sha": "abc1234567890abc1234567890abc1234567890ab",
                "ref": "feature/test",
            },
            "base": {
                "sha": "def1234567890def1234567890def1234567890de",
                "ref": "main",
            },
        },
        "repository": {
            "id": 123456,
            "name": "testrepo",
            "full_name": "testorg/testrepo",
            "clone_url": "https://github.com/testorg/testrepo.git",
            "default_branch": "main",
            "language": "Python",
        },
        "installation": {"id": 999},
    }


class TestWebhookSignatureVerification:
    """Test HMAC signature validation."""

    def test_valid_signature_accepted(self, test_app):
        body = json.dumps(_make_pr_payload()).encode()
        sig = _sign_payload(body, TEST_WEBHOOK_SECRET)

        with patch("apps.webhook_api.routes.webhook.TaskQueue.enqueue", new_callable=AsyncMock) as mock_enqueue:
            mock_enqueue.return_value = "test-job-001"
            with TestClient(test_app) as client:
                response = client.post(
                    "/webhooks/github",
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Hub-Signature-256": sig,
                        "X-GitHub-Event": "pull_request",
                        "X-GitHub-Delivery": "test-delivery-001",
                    },
                )
            # 202 or 200 both acceptable for async queue response
            assert response.status_code in (200, 202)

    def test_missing_signature_returns_401(self, test_app):
        body = json.dumps(_make_pr_payload()).encode()
        with TestClient(test_app) as client:
            response = client.post(
                "/webhooks/github",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-GitHub-Event": "pull_request",
                },
            )
        assert response.status_code == 401

    def test_invalid_signature_returns_403(self, test_app):
        body = json.dumps(_make_pr_payload()).encode()
        with TestClient(test_app) as client:
            response = client.post(
                "/webhooks/github",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": "sha256=invalidsignature",
                    "X-GitHub-Event": "pull_request",
                },
            )
        assert response.status_code == 403


class TestWebhookEventFiltering:
    """Test that only actionable PR events are processed."""

    def test_non_pr_event_is_ignored(self, test_app):
        body = json.dumps({"action": "created"}).encode()
        sig = _sign_payload(body, TEST_WEBHOOK_SECRET)
        with TestClient(test_app) as client:
            response = client.post(
                "/webhooks/github",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Event": "push",  # Not a PR event
                },
            )
        assert response.status_code in (200, 202)
        data = response.json()
        assert "ignored" in data.get("message", "").lower() or data.get("job_id") == "ignored"

    def test_pr_closed_action_is_ignored(self, test_app):
        body = json.dumps(_make_pr_payload(action="closed")).encode()
        sig = _sign_payload(body, TEST_WEBHOOK_SECRET)
        with TestClient(test_app) as client:
            response = client.post(
                "/webhooks/github",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Event": "pull_request",
                },
            )
        data = response.json()
        assert data.get("job_id") == "ignored"


class TestHealthCheck:
    def test_health_endpoint_reachable(self, test_app):
        with patch("apps.webhook_api.main.ping_redis", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = True
            with TestClient(test_app) as client:
                response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data