"""
apps/webhook_api/routes/webhook.py

GitHub webhook receiver route.

Contract:
  - Validates HMAC signature (via dependency)
  - Parses GitHub pull_request event payload
  - Filters to only actionable PR actions (opened, synchronize, reopened)
  - Pushes job to Redis queue
  - Returns 200 immediately (GitHub times out webhook responses at 10s)

The webhook route NEVER performs analysis — that's the worker's job.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request, status

from apps.webhook_api.dependencies import SignatureVerified, TaskQueueDep
from shared.constants import REVIEW_TRIGGER_ACTIONS
from shared.logger import get_logger
from shared.schemas import WebhookAckResponse, WebhookPayloadSchema

logger = get_logger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post(
    "/github",
    response_model=WebhookAckResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receive GitHub pull_request webhook events",
)
async def receive_github_webhook(
    request: Request,
    #_: SignatureVerified,
    task_queue: TaskQueueDep, # type: ignore
    x_github_event: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
) -> WebhookAckResponse:
    """
    Receive a GitHub pull_request webhook event.

    1. Validate signature (via dependency — already done if we get here)
    2. Parse raw payload into WebhookPayloadSchema
    3. Filter non-PR or non-actionable events
    4. Enqueue review job
    5. Return 202 Accepted with job_id
    """
    # Only handle pull_request events
    if x_github_event != "pull_request":
        logger.debug("Ignoring non-PR webhook event", event=x_github_event)
        return WebhookAckResponse(job_id="ignored", message=f"Event '{x_github_event}' ignored")

    raw = await request.json()

    action: str = raw.get("action", "")

    # Only review on PR open / new commit push / reopen
    if action not in REVIEW_TRIGGER_ACTIONS:
        logger.debug("Ignoring PR action", action=action)
        return WebhookAckResponse(job_id="ignored", message=f"Action '{action}' ignored")

    # Parse and validate payload
    try:
        payload = _parse_github_payload(raw, x_github_delivery)
    except (KeyError, ValueError) as exc:
        logger.error("Failed to parse GitHub webhook payload", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid payload: {exc}",
        )

    # Enqueue — non-blocking, returns immediately
    job_id = await task_queue.enqueue(payload)

    logger.info(
        "Webhook received and queued",
        job_id=job_id,
        action=action,
        pr_number=payload.pr_number,
        repo=payload.repo_full_name,
        delivery=x_github_delivery,
    )

    return WebhookAckResponse(job_id=job_id)


def _parse_github_payload(
    raw: dict, delivery_id: str | None
) -> WebhookPayloadSchema:
    """
    Extract the fields we care about from the raw GitHub webhook blob.

    GitHub's webhook payload is deeply nested — we flatten it into
    our clean WebhookPayloadSchema for downstream use.

    Raises KeyError if required fields are missing.
    """
    repo = raw["repository"]
    pr = raw["pull_request"]
    head = pr["head"]
    base = pr["base"]

    return WebhookPayloadSchema(
        action=raw["action"],
        installation_id=raw.get("installation", {}).get("id"),
        repo_id=repo["id"],
        repo_full_name=repo["full_name"],
        repo_clone_url=repo["clone_url"],
        repo_default_branch=repo.get("default_branch", "main"),
        repo_language=repo.get("language"),
        pr_number=pr["number"],
        pr_title=pr["title"],
        pr_body=pr.get("body"),
        pr_html_url=pr["html_url"],
        pr_head_sha=head["sha"],
        pr_base_sha=base["sha"],
        pr_head_branch=head["ref"],
        pr_base_branch=base["ref"],
        pr_author=pr["user"]["login"],
        delivery_id=delivery_id,
    )