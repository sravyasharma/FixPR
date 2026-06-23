"""
tests/test_queue.py

Unit tests for TaskQueue serialization and job lifecycle.
Uses fakeredis to avoid needing a real Redis instance.
"""

from __future__ import annotations

import json
import pytest
import pytest_asyncio

from shared.schemas import ReviewJobSchema, WebhookPayloadSchema


class TestReviewJobSerialization:
    """Ensure ReviewJobSchema round-trips through JSON cleanly."""

    def test_job_serializes_to_json(self, sample_job):
        raw = sample_job.model_dump_json()
        assert isinstance(raw, str)
        data = json.loads(raw)
        assert data["job_id"] == "test-job-001"
        assert data["schema_version"] == "1.0"
        assert data["payload"]["pr_number"] == 42

    def test_job_deserializes_from_json(self, sample_job):
        raw = sample_job.model_dump_json()
        restored = ReviewJobSchema.model_validate_json(raw)
        assert restored.job_id == sample_job.job_id
        assert restored.payload.pr_number == sample_job.payload.pr_number
        assert restored.payload.repo_full_name == sample_job.payload.repo_full_name

    def test_job_is_immutable(self, sample_job):
        """Frozen Pydantic models raise ValidationError on attribute set."""
        with pytest.raises(Exception):
            sample_job.job_id = "modified"

    def test_retry_count_defaults_to_zero(self, sample_job):
        assert sample_job.retry_count == 0


"""
tests/test_repositories.py

Integration tests for storage repositories using SQLite in-memory DB.
"""


class TestRepositoryRepo:
    """Tests for RepositoryRepo."""

    @pytest.mark.asyncio
    async def test_get_or_create_new_repo(self, db_session, sample_payload):
        from storage.repositories import RepositoryRepo

        repo = RepositoryRepo(db_session)
        db_repo = await repo.get_or_create(sample_payload)

        assert db_repo.full_name == "testorg/testrepo"
        assert db_repo.github_repo_id == 123456
        assert db_repo.default_branch == "main"

    @pytest.mark.asyncio
    async def test_get_or_create_idempotent(self, db_session, sample_payload):
        from storage.repositories import RepositoryRepo

        repo = RepositoryRepo(db_session)
        first = await repo.get_or_create(sample_payload)
        second = await repo.get_or_create(sample_payload)

        assert str(first.id) == str(second.id)

    @pytest.mark.asyncio
    async def test_get_by_full_name(self, db_session, sample_payload):
        from storage.repositories import RepositoryRepo

        repo = RepositoryRepo(db_session)
        await repo.get_or_create(sample_payload)

        found = await repo.get_by_full_name("testorg/testrepo")
        assert found is not None
        assert found.github_repo_id == 123456

        not_found = await repo.get_by_full_name("nonexistent/repo")
        assert not_found is None


class TestReviewRepo:
    """Tests for ReviewRepo."""

    @pytest.mark.asyncio
    async def test_create_review(self, db_session, sample_payload):
        import uuid
        from storage.repositories import PullRequestRepo, RepositoryRepo, ReviewRepo

        # Create repo and PR first
        repo_repo = RepositoryRepo(db_session)
        pr_repo = PullRequestRepo(db_session)
        review_repo = ReviewRepo(db_session)

        db_repo = await repo_repo.get_or_create(sample_payload)
        db_pr = await pr_repo.get_or_create(sample_payload, db_repo.id)

        review_id = str(uuid.uuid4())
        review = await review_repo.create(
            pr_id=db_pr.id,
            job_id="test-job-001",
            review_id=review_id,
        )

        assert review.job_id == "test-job-001"
        assert review.status == "PENDING"

    @pytest.mark.asyncio
    async def test_complete_review(self, db_session, sample_payload):
        import uuid
        from shared.schemas import MergedReviewResult
        from storage.repositories import PullRequestRepo, RepositoryRepo, ReviewRepo

        repo_repo = RepositoryRepo(db_session)
        pr_repo = PullRequestRepo(db_session)
        review_repo = ReviewRepo(db_session)

        db_repo = await repo_repo.get_or_create(sample_payload)
        db_pr = await pr_repo.get_or_create(sample_payload, db_repo.id)

        review_id = str(uuid.uuid4())
        await review_repo.create(
            pr_id=db_pr.id,
            job_id="test-job-002",
            review_id=review_id,
        )

        merged = MergedReviewResult(
            findings=[],
            critical_count=1,
            high_count=2,
            markdown_report="# Report",
            duration_seconds=5.2,
        )

        await review_repo.complete(review_id, merged)

        updated = await review_repo.get_by_id(review_id)
        assert updated.status == "COMPLETED"
        assert updated.critical_count == 1
        assert updated.high_count == 2


class TestFindingRepo:
    """Tests for FindingRepo bulk insert."""

    @pytest.mark.asyncio
    async def test_bulk_insert_findings(self, db_session, sample_payload, critical_finding, high_finding):
        import uuid
        from storage.repositories import FindingRepo, PullRequestRepo, RepositoryRepo, ReviewRepo

        repo_repo = RepositoryRepo(db_session)
        pr_repo = PullRequestRepo(db_session)
        review_repo = ReviewRepo(db_session)
        finding_repo = FindingRepo(db_session)

        db_repo = await repo_repo.get_or_create(sample_payload)
        db_pr = await pr_repo.get_or_create(sample_payload, db_repo.id)
        review_id = str(uuid.uuid4())
        await review_repo.create(pr_id=db_pr.id, job_id="test-job-003", review_id=review_id)

        inserted = await finding_repo.bulk_insert(
            review_id, [critical_finding, high_finding]
        )

        assert len(inserted) == 2

        queried = await finding_repo.get_by_review(review_id)
        assert len(queried) == 2