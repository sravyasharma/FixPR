"""
apps/webhook_api/dependencies.py

FastAPI dependency injectors for the webhook API.

Provides:
  - verify_github_signature()  — HMAC-SHA256 webhook verification
  - get_task_queue()           — TaskQueue singleton
  - get_db()                   — AsyncSession for request scope

Architecture role:
  - Injected into route handlers via FastAPI Depends()
  - Keeps route handlers thin — all setup/teardown here
"""

from __future__ import annotations

from typing import Annotated, AsyncIterator

from fastapi import Depends, Header, HTTPException, Request, status

from queue.task_queue import TaskQueue # type: ignore
from shared.config import get_settings
from shared.logger import get_logger
from storage.db import get_db_session

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# GitHub Signature Verification
# ------------------------------------------------------------------ #


async def verify_github_signature(
    request: Request,
    x_hub_signature_256: Annotated[str | None, Header()] = None,
) -> None:
    """
    Verify the GitHub webhook HMAC-SHA256 signature.

    GitHub sends: X-Hub-Signature-256: sha256=<hex_digest>
    We compute: HMAC-SHA256(secret, body) and compare.

    Raises 401 if signature is missing or invalid.
    Raises 403 if the computed signature doesn't match.

    This dependency must run BEFORE any body parsing so we can read
    the raw bytes (parsed body can't be re-hashed).
    """
    import hashlib
    import hmac

    settings = get_settings()
    body = await request.body()

    if not x_hub_signature_256:
        logger.warning("Webhook request missing signature header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Hub-Signature-256 header",
        )

    expected = "sha256=" + hmac.new(
        key=settings.github_webhook_secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, x_hub_signature_256):
        logger.warning(
            "Webhook signature mismatch",
            received=x_hub_signature_256[:20] + "...",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid webhook signature",
        )


# ------------------------------------------------------------------ #
# Queue
# ------------------------------------------------------------------ #

_task_queue: TaskQueue | None = None


def get_task_queue() -> TaskQueue:
    """Return the singleton TaskQueue. Thread-safe singleton."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


# Type alias for injection
TaskQueueDep = Annotated[TaskQueue, Depends(get_task_queue)]
SignatureVerified = Annotated[None, Depends(verify_github_signature)]