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

import hashlib
import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from task_queue.task_queue import TaskQueue
from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# GitHub Signature Verification
# ------------------------------------------------------------------ #

async def verify_github_signature(
    request: Request,
    x_hub_signature_256: Annotated[str | None, Header()] = None,
) -> None:
    """Verify the GitHub webhook HMAC-SHA256 signature."""
    if not x_hub_signature_256:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Hub-Signature-256 header",
        )

    body = await request.body()
    expected = "sha256=" + hmac.new(
        key=get_settings().github_webhook_secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, x_hub_signature_256):
        logger.warning("Webhook signature mismatch")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid webhook signature",
        )


# ------------------------------------------------------------------ #
# Queue
# ------------------------------------------------------------------ #

_task_queue: TaskQueue | None = None


def get_task_queue() -> TaskQueue:
    """Return the singleton TaskQueue."""
    global _task_queue

    if _task_queue is None:
        _task_queue = TaskQueue()

    return _task_queue


# ------------------------------------------------------------------ #
# Dependency Aliases
# ------------------------------------------------------------------ #

TaskQueueDep = Annotated[
    TaskQueue,
    Depends(get_task_queue),
]

SignatureVerified = Annotated[
    None,
    Depends(verify_github_signature),
]
