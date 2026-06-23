"""
apps/worker/worker.py

Async worker — consumes jobs from Redis and runs the LangGraph review pipeline.

Architecture:
  - One worker process runs N concurrent coroutines (WORKER_CONCURRENCY)
  - Each coroutine blocks on Redis dequeue, processes one job, loops
  - Semaphore ensures we never exceed max concurrency
  - Graceful shutdown on SIGTERM/SIGINT

This is the CONSUMER side. The webhook API is the PRODUCER.

Start with:
    python -m apps.worker.worker

Or via Docker:
    docker-compose up worker
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from contextlib import asynccontextmanager

from agents.orchestrator import get_review_graph
from task_queue.redis_client import close_redis, ping_redis 
from task_queue.task_queue import TaskQueue 
from shared.config import get_settings
from shared.constants import ReviewStatus
from shared.logger import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from shared.schemas import PipelineState, ReviewJobSchema
from storage.db import close_engine, create_all_tables, get_db_session
from storage.repositories import PullRequestRepo, RepositoryRepo, ReviewRepo
from tracing.langsmith import configure_tracing, get_tracer_config

logger = get_logger(__name__)

# Global shutdown flag — set by signal handlers
_shutdown = False


def _handle_shutdown(sig, frame) -> None:
    """Signal handler for graceful shutdown."""
    global _shutdown
    logger.info("Shutdown signal received", signal=sig)
    _shutdown = True


# ------------------------------------------------------------------ #
# Job Processor
# ------------------------------------------------------------------ #


async def process_job(
    job: ReviewJobSchema,
    task_queue: TaskQueue,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Process a single review job through the full LangGraph pipeline.

    Steps:
      1. Bind structured logging context for this job
      2. Create review record in DB
      3. Run the compiled LangGraph pipeline
      4. ACK job on success, requeue on recoverable failure
      5. Dead-letter on unrecoverable failure
    """
    async with semaphore:
        payload = job.payload
        job_start = time.monotonic()

        # Bind context for all log calls in this job's scope
        bind_request_context(
            job_id=job.job_id,
            pr_number=payload.pr_number,
            repo=payload.repo_full_name,
        )

        logger.info(
            "Processing job",
            action=payload.action,
            retry_count=job.retry_count,
        )

        review_id: str | None = None

        try:
            # Create the review record (status=PENDING)
            async with get_db_session() as session:
                repo_repo = RepositoryRepo(session)
                pr_repo = PullRequestRepo(session)
                review_repo = ReviewRepo(session)

                db_repo = await repo_repo.get_or_create(payload)
                db_pr = await pr_repo.get_or_create(payload, db_repo.id)
                review = await review_repo.create(
                    pr_id=db_pr.id,
                    job_id=job.job_id,
                    review_id=job.job_id,  # Use job_id as review_id for traceability
                )
                review_id = str(review.id)

            # Build initial pipeline state
            initial_state = PipelineState(
                job=job,
                review_id=review_id or job.job_id,
            )

            # Run the LangGraph pipeline
            graph = get_review_graph()
            tracer_config = get_tracer_config(f"review-pr{payload.pr_number}-{job.job_id[:8]}")

            final_state_dict = await graph.ainvoke(
                initial_state,
                config=tracer_config,
            )

            duration = time.monotonic() - job_start
            logger.info(
                "Job completed successfully",
                duration=f"{duration:.2f}s",
                review_id=review_id,
            )

            # ACK — remove from in-flight list
            await task_queue.ack(job)

        except Exception as exc:
            duration = time.monotonic() - job_start
            logger.error(
                "Job failed",
                error=str(exc),
                duration=f"{duration:.2f}s",
                exc_info=True,
            )

            # Mark review as failed in DB
            if review_id:
                try:
                    async with get_db_session() as session:
                        review_repo = ReviewRepo(session)
                        await review_repo.fail(review_id, str(exc)[:2048])
                except Exception as db_exc:
                    logger.error("Failed to mark review as failed in DB", error=str(db_exc))

            # Decide: requeue (transient) or dead-letter (permanent)
            if _is_transient_error(exc) and job.retry_count < 3:
                logger.warning("Requeueing job for retry", retry_count=job.retry_count + 1)
                await task_queue.requeue(job)
            else:
                await task_queue.dead_letter(job, str(exc))

        finally:
            clear_request_context()


def _is_transient_error(exc: Exception) -> bool:
    """
    Determine if an error is transient (worth retrying) or permanent.

    Transient: network errors, temporary DB unavailability, rate limits
    Permanent: parsing errors, missing files, auth failures
    """
    transient_types = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    transient_messages = [
        "rate limit",
        "timeout",
        "connection reset",
        "temporarily unavailable",
        "503",
        "502",
    ]

    if isinstance(exc, transient_types):
        return True

    error_str = str(exc).lower()
    return any(msg in error_str for msg in transient_messages)


# ------------------------------------------------------------------ #
# Worker Coroutine
# ------------------------------------------------------------------ #


async def worker_loop(
    worker_id: int,
    task_queue: TaskQueue,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Single worker coroutine — polls Redis and processes jobs.

    Runs until the global _shutdown flag is set.
    """
    logger.info("Worker started", worker_id=worker_id)

    while not _shutdown:
        try:
            job = await task_queue.dequeue(timeout=5)

            if job is None:
                # No job in queue — loop and try again
                continue

            # Schedule job processing (non-blocking — semaphore controls concurrency)
            asyncio.create_task(
                process_job(job, task_queue, semaphore),
                name=f"job-{job.job_id[:8]}",
            )

        except asyncio.CancelledError:
            logger.info("Worker coroutine cancelled", worker_id=worker_id)
            break
        except Exception as exc:
            logger.error(
                "Worker loop error",
                worker_id=worker_id,
                error=str(exc),
                exc_info=True,
            )
            # Brief pause to avoid tight error loops
            await asyncio.sleep(2)

    logger.info("Worker stopped", worker_id=worker_id)


# ------------------------------------------------------------------ #
# Entrypoint
# ------------------------------------------------------------------ #


async def main() -> None:
    """
    Start the worker process.

    1. Configure logging and tracing
    2. Verify DB and Redis connectivity
    3. Ensure DB schema exists
    4. Spin up N worker coroutines
    5. Wait for shutdown signal
    """
    configure_logging()
    configure_tracing()

    settings = get_settings()

    logger.info(
        "Starting worker",
        concurrency=settings.worker_concurrency,
        queue=settings.redis_queue_name,
        env=settings.app_env,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    # Connectivity checks
    if not await ping_redis():
        logger.error("Cannot connect to Redis — exiting")
        sys.exit(1)

    await create_all_tables()

    task_queue = TaskQueue()
    semaphore = asyncio.Semaphore(settings.worker_concurrency)

    # Pre-compile the LangGraph pipeline (one-time JIT compilation)
    logger.info("Compiling LangGraph review pipeline...")
    _ = get_review_graph()
    logger.info("Pipeline compiled and ready")

    # Start N worker coroutines
    workers = [
        asyncio.create_task(
            worker_loop(i, task_queue, semaphore),
            name=f"worker-{i}",
        )
        for i in range(settings.worker_concurrency)
    ]

    logger.info(
        "All workers running",
        count=settings.worker_concurrency,
    )

    # Wait for all workers (they run until _shutdown=True)
    try:
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down — draining workers...")
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        await close_engine()
        await close_redis()
        logger.info("Worker shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())