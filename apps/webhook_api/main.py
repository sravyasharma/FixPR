"""
apps/webhook_api/main.py

FastAPI application entry point for the AI Code Review Platform webhook API.

Responsibilities:
  - Create the FastAPI app with lifespan context (startup/shutdown)
  - Register all routers
  - Configure middleware (CORS, request logging, error handling)
  - Health check endpoint

This process is the PRODUCER — it only receives webhooks and enqueues jobs.
It never runs analysis directly.

Start with:
    uvicorn apps.webhook_api.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.webhook_api.routes.reviews import router as reviews_router
from apps.webhook_api.routes.webhook import router as webhook_router
from queue.redis_client import close_redis, ping_redis # type: ignore
from shared.config import get_settings
from shared.logger import configure_logging, get_logger
from storage.db import close_engine, create_all_tables
from tracing.langsmith import configure_tracing

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Lifespan
# ------------------------------------------------------------------ #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Startup:
      - Configure logging and tracing
      - Ensure database tables exist
      - Verify Redis connectivity

    Shutdown:
      - Close DB connection pool
      - Close Redis client
    """
    # --- Startup ---
    configure_logging()
    configure_tracing()

    settings = get_settings()
    logger.info(
        "Webhook API starting",
        env=settings.app_env,
        app=settings.app_name,
    )

    # Ensure DB schema exists (idempotent)
    await create_all_tables()

    # Verify Redis is reachable
    if not await ping_redis():
        logger.error("Redis is not reachable at startup — jobs cannot be queued")
        # Don't crash — let health check surface this
    else:
        logger.info("Redis connection verified")

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Webhook API shutting down")
    await close_engine()
    await close_redis()
    logger.info("Shutdown complete")


# ------------------------------------------------------------------ #
# App Factory
# ------------------------------------------------------------------ #


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Code Review Platform — Webhook API",
        description=(
            "Receives GitHub pull_request webhook events and queues them "
            "for automated AI-powered code review."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    # ---- Middleware ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ---- Exception Handlers ----
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )

    # ---- Routes ----
    app.include_router(webhook_router)
    app.include_router(reviews_router)

    # ---- Health Check ----
    @app.get("/health", tags=["health"], include_in_schema=False)
    async def health_check() -> dict:
        redis_ok = await ping_redis()
        return {
            "status": "healthy" if redis_ok else "degraded",
            "redis": "ok" if redis_ok else "unreachable",
            "env": settings.app_env,
        }

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"service": "github-ai-review-agent", "docs": "/docs"}

    return app


app = create_app()