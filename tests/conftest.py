"""
tests/conftest.py

Shared pytest fixtures for the AI Code Review Platform test suite.

Provides:
  - async event loop (module-scoped for performance)
  - in-memory SQLite engine for DB tests (no Postgres required)
  - sample WebhookPayloadSchema for unit tests
  - mock PRDiffSchema with realistic file changes
  - pre-built FindingSchema factories
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from shared.constants import AnalysisSource, ConfidenceLevel, Severity
from shared.schemas import (
    FileChangeSchema,
    FindingSchema,
    PRDiffSchema,
    ReviewJobSchema,
    WebhookPayloadSchema,
)
from storage.models import Base


# ------------------------------------------------------------------ #
# Event Loop
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ------------------------------------------------------------------ #
# Test Database (SQLite in-memory)
# ------------------------------------------------------------------ #


@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create a fresh async SQLite in-memory engine per test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine) -> AsyncIterator[AsyncSession]:
    """Provide a test DB session that rolls back after each test."""
    factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with factory() as session:
        yield session
        await session.rollback()


# ------------------------------------------------------------------ #
# Sample Data Factories
# ------------------------------------------------------------------ #


@pytest.fixture
def sample_payload() -> WebhookPayloadSchema:
    return WebhookPayloadSchema(
        action="opened",
        repo_id=123456,
        repo_full_name="testorg/testrepo",
        repo_clone_url="https://github.com/testorg/testrepo.git",
        repo_default_branch="main",
        repo_language="Python",
        pr_number=42,
        pr_title="Add user authentication feature",
        pr_body="This PR adds JWT-based authentication.",
        pr_html_url="https://github.com/testorg/testrepo/pull/42",
        pr_head_sha="abc1234567890abc1234567890abc1234567890ab",
        pr_base_sha="def1234567890def1234567890def1234567890de",
        pr_head_branch="feature/auth",
        pr_base_branch="main",
        pr_author="testuser",
        delivery_id="test-delivery-001",
    )


@pytest.fixture
def sample_job(sample_payload) -> ReviewJobSchema:
    return ReviewJobSchema(
        job_id="test-job-001",
        payload=sample_payload,
    )


@pytest.fixture
def sample_python_file_change() -> FileChangeSchema:
    return FileChangeSchema(
        file_path="src/auth/jwt_handler.py",
        change_type="added",
        patch="""@@ -0,0 +1,20 @@
+import jwt
+import os
+
+SECRET_KEY = "hardcoded_secret_do_not_use"
+
+def create_token(user_id: int) -> str:
+    payload = {"user_id": user_id}
+    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
+
+def verify_token(token: str) -> dict:
+    try:
+        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
+    except Exception:
+        return {}
""",
        additions=14,
        deletions=0,
        language="python",
    )


@pytest.fixture
def sample_diff(sample_python_file_change) -> PRDiffSchema:
    return PRDiffSchema(
        repo_full_name="testorg/testrepo",
        pr_number=42,
        head_sha="abc1234567890abc1234567890abc1234567890ab",
        base_sha="def1234567890def1234567890def1234567890de",
        local_repo_path="/tmp/test_repo",
        changed_files=[sample_python_file_change],
    )


def make_finding(
    source: AnalysisSource = AnalysisSource.PYLINT,
    severity: Severity = Severity.HIGH,
    file_path: str = "src/auth/jwt_handler.py",
    line_number: int = 4,
    issue: str = "Hardcoded secret key detected",
    suggestion: str = "Use environment variable for secret key",
    confidence_score: float = 0.90,
) -> FindingSchema:
    return FindingSchema(
        source=source,
        file_path=file_path,
        line_number=line_number,
        severity=severity,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=confidence_score,
        issue=issue,
        suggestion=suggestion,
    )


@pytest.fixture
def critical_finding() -> FindingSchema:
    return make_finding(
        source=AnalysisSource.BANDIT,
        severity=Severity.CRITICAL,
        issue="[B105] Hardcoded password string detected",
        suggestion="Store secrets in environment variables or a secrets manager",
    )


@pytest.fixture
def high_finding() -> FindingSchema:
    return make_finding(
        source=AnalysisSource.SEMGREP,
        severity=Severity.HIGH,
        issue="[jwt.decode] Missing signature verification options",
        suggestion="Pass options={'verify_exp': True} to jwt.decode()",
        line_number=11,
    )


@pytest.fixture
def medium_finding() -> FindingSchema:
    return make_finding(
        source=AnalysisSource.LLM_REVIEW,
        severity=Severity.MEDIUM,
        issue="Empty dict returned on token verification failure may hide errors",
        suggestion="Raise a specific AuthenticationError instead of returning {}",
        line_number=13,
        confidence_score=0.75,
    )