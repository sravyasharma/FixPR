"""
queue/task_queue.py

Redis-backed task queue with reliable delivery semantics.

Uses BRPOPLPUSH (or BLMOVE in Redis 6.2+) pattern:
  - Jobs pushed to "pr_review_jobs" (pending list)
  - Worker atomically moves job to "pr_review_jobs:processing" (in-flight list)
  - On completion, job removed from processing list
  - On worker crash, jobs remain in processing list for requeue

Architecture role:
  - Webhook API uses enqueue() → fire and forget
  - Worker uses dequeue() → blocking pop with processing guarantee
  - Worker calls ack() on success, requeue() on transient failure
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from shared.config import get_settings
from shared.logger import get_logger
from shared.schemas import ReviewJobSchema, WebhookPayloadSchema
from task_queue.redis_client import get_redis_client 

logger = get_logger(__name__)


def _queue_name() -> str:
    return get_settings().redis_queue_name


def _processing_queue_name() -> str:
    return f"{_queue_name()}:processing"


def _dead_letter_queue_name() -> str:
    return f"{_queue_name()}:dead_letter"


class TaskQueue:
    """
    Async task queue backed by Redis lists.

    Producer (webhook API):
        queue = TaskQueue()
        job_id = await queue.enqueue(payload)

    Consumer (worker):
        queue = TaskQueue()
        while True:
            job = await queue.dequeue(timeout=5)
            if job:
                try:
                    await process(job)
                    await queue.ack(job)
                except RecoverableError:
                    await queue.requeue(job)
                except Exception:
                    await queue.dead_letter(job, error)
    """

    def __init__(self) -> None:
        self._redis = get_redis_client()
        self._settings = get_settings()

    async def enqueue(self, payload: WebhookPayloadSchema) -> str:
        """
        Serialize a ReviewJob and push it to the pending list.

        Returns the job_id so the webhook can return it in the response.
        """
        job = ReviewJobSchema(payload=payload)
        job_bytes = job.model_dump_json().encode("utf-8")

        queue = _queue_name()
        await self._redis.lpush(queue, job_bytes)

        logger.info(
            "Job enqueued",
            job_id=job.job_id,
            pr_number=payload.pr_number,
            repo=payload.repo_full_name,
            queue=queue,
        )
        return job.job_id

    async def dequeue(self, timeout: int = 5) -> tuple[ReviewJobSchema | None, bytes | None]:
        """
        Blocking pop from pending → atomically push to processing.

        Returns None on timeout (expected — caller should loop).
        """
        #pending_q = _queue_name()
        #processing_q = _processing_queue_name()

        # BRPOPLPUSH: atomically move item from tail of pending to head of processing
        #raw = await self._redis.brpoplpush(pending_q, processing_q, timeout=timeout)
        raw = await self._redis.brpoplpush(...)
        if raw is None:
            return None,None
            
        
        try:
            data = json.loads(raw)
            job = ReviewJobSchema.model_validate(data)
            logger.info(
                "Job dequeued",
                job_id=job.job_id,
                pr_number=job.payload.pr_number,
                repo=job.payload.repo_full_name,
            )
            return job
        except Exception as exc:
            logger.error(
                "Failed to deserialize job from Redis",
                error=str(exc),
                raw=raw[:200] if raw else None,
            )
            # Move malformed job to dead letter
            await self._redis.lrem(_processing_queue_name(), 1, raw)
            await self._redis.lpush(_dead_letter_queue_name(), raw)
            return None

    async def ack(self, job: ReviewJobSchema, raw_bytes: bytes) -> None:
        """
        Acknowledge successful processing.

        Removes the job from the in-flight processing list.
        """
        #job_bytes = job.model_dump_json().encode("utf-8")
        removed = await self._redis.lrem(_processing_queue_name(), 1, raw_bytes)
        logger.info("Job acknowledged", job_id=job.job_id, removed=removed)

    async def requeue(self, job: ReviewJobSchema) -> None:
        """
        Requeue a job after a transient failure.

        Increments retry_count. After MAX_RETRIES, moves to dead letter.
        """
        from shared.constants import MAX_JOB_RETRY_COUNT

        # Remove from processing
        original_bytes = job.model_dump_json().encode("utf-8")
        await self._redis.lrem(_processing_queue_name(), 1, original_bytes)

        new_retry_count = job.retry_count + 1
        if new_retry_count > MAX_JOB_RETRY_COUNT:
            logger.error(
                "Job exceeded max retries — moving to dead letter",
                job_id=job.job_id,
                retry_count=new_retry_count,
            )
            await self._redis.lpush(_dead_letter_queue_name(), original_bytes)
            return

        # Rebuild job with incremented retry count (frozen model — rebuild)
        retried_job = ReviewJobSchema(
            job_id=job.job_id,
            schema_version=job.schema_version,
            created_at=job.created_at,
            retry_count=new_retry_count,
            payload=job.payload,
        )
        retried_bytes = retried_job.model_dump_json().encode("utf-8")
        await self._redis.lpush(_queue_name(), retried_bytes)

        logger.warning(
            "Job requeued for retry",
            job_id=job.job_id,
            retry_count=new_retry_count,
        )

    async def dead_letter(self, job: ReviewJobSchema, error: str) -> None:
        """Move a permanently failed job to the dead letter queue."""
        job_bytes = job.model_dump_json().encode("utf-8")
        await self._redis.lrem(_processing_queue_name(), 1, job_bytes)

        # Store with error metadata
        dlq_entry = {
            "job": json.loads(job_bytes),
            "error": error[:1000],
            "dead_lettered_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        await self._redis.lpush(
            _dead_letter_queue_name(), json.dumps(dlq_entry).encode("utf-8")
        )
        logger.error(
            "Job moved to dead letter queue",
            job_id=job.job_id,
            error=error[:200],
        )

    async def queue_length(self) -> int:
        """Return the number of pending jobs."""
        return await self._redis.llen(_queue_name())

    async def processing_length(self) -> int:
        """Return the number of in-flight jobs."""
        return await self._redis.llen(_processing_queue_name())

    async def dead_letter_length(self) -> int:
        """Return the number of dead-lettered jobs."""
        return await self._redis.llen(_dead_letter_queue_name())