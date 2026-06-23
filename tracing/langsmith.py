"""
tracing/langsmith.py

LangSmith tracing configuration for the AI Code Review Platform.

LangSmith automatically traces:
  - LangGraph node execution (latency, inputs, outputs)
  - LLM calls (prompt, completion, token counts, cost)
  - Failures and retries with full context

Architecture role:
  - Called once at process startup (main.py / worker.py)
  - Sets environment variables that LangChain/LangGraph read automatically
  - No explicit instrumentation needed in agent code
"""

from __future__ import annotations

import os

from shared.config import get_settings
from shared.logger import get_logger

logger = get_logger(__name__)


def configure_tracing() -> None:
    """
    Configure LangSmith tracing by setting environment variables.

    LangChain reads LANGCHAIN_* env vars at import time, so this must
    be called before any langchain/langgraph imports.

    When tracing is disabled (default in dev), this is a no-op.
    """
    settings = get_settings()

    if not settings.langchain_tracing_v2 or not settings.langchain_api_key:
        logger.info("LangSmith tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

    logger.info(
        "LangSmith tracing enabled",
        project=settings.langchain_project,
        endpoint=settings.langchain_endpoint,
    )


def get_tracer_config(run_name: str) -> dict:
    """
    Return a LangGraph run config with tracing metadata.

    Pass this as the `config` parameter to graph.ainvoke():
        config = get_tracer_config(f"review-{job_id}")
        await graph.ainvoke(state, config=config)

    This tags the trace with the job-specific run name so you can
    filter by PR or job in the LangSmith dashboard.
    """
    settings = get_settings()
    return {
        "run_name": run_name,
        "metadata": {
            "project": settings.langchain_project,
            "app": settings.app_name,
            "env": settings.app_env,
        },
        "tags": [settings.app_env, "code-review"],
    }