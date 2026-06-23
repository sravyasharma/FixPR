"""
shared/logger.py

Structured JSON logging for the AI Code Review Platform.

Uses structlog for:
- JSON output in production (parseable by Datadog, CloudWatch, etc.)
- Human-readable colored output in development
- Context binding (bind PR number, repo, job ID once, log everywhere)
- Automatic exception formatting with tracebacks

Architecture role:
- Every module calls get_logger(__name__) at module level
- Middleware binds request-scoped context (job_id, pr_number, repo)
- Worker binds job context at job start
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from shared.config import get_settings


def _add_log_level(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add the log level name to every event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def _drop_color_message_key(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Remove uvicorn's 'color_message' key — we don't need ANSI in structured logs."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging() -> None:
    """
    Configure structlog and stdlib logging.

    Call once at application startup (main.py / worker.py).
    Subsequent calls are no-ops because structlog stores config globally.
    """
    settings = get_settings()
    is_production = settings.app_env == "production"

    # Shared processors — applied in all environments
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        _add_log_level,
        _drop_color_message_key,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if is_production:
        # JSON output for log aggregators
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty colored output for local development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, sqlalchemy, etc.) through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.log_level),
    )

    # Quiet down noisy libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a named structlog logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("PR received", pr_number=42, repo="org/repo")

    Context binding (survives across async tasks in same contextvars scope):
        structlog.contextvars.bind_contextvars(job_id="abc123", pr_number=42)
    """
    return structlog.get_logger(name)


def bind_request_context(**kwargs: Any) -> None:
    """
    Bind key-value pairs to the current async task's log context.

    Call in middleware or at job start. All subsequent log calls in this
    contextvars scope will include these fields automatically.

    Example:
        bind_request_context(job_id=job.id, pr_number=pr.number, repo=repo.full_name)
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """Clear all bound context. Call at end of request/job processing."""
    structlog.contextvars.clear_contextvars()