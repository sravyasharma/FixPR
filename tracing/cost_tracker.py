"""
tracing/cost_tracker.py

LLM cost tracking — records token usage and USD cost to PostgreSQL.

Architecture role:
  - Called by orchestrator node_persist_results after LLM calls
  - Inserts LLMUsageModel rows via LLMUsageRepo
  - Provides aggregate cost queries for billing dashboards
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from shared.constants import MODEL_COST_MAP
from shared.logger import get_logger
from shared.schemas import LLMUsageSchema
from storage.repositories import LLMUsageRepo

logger = get_logger(__name__)


class CostTracker:
    """
    Records LLM API call costs to the database.

    Usage:
        tracker = CostTracker()
        await tracker.record(
            session=session,
            review_id="...",
            agent_name="llm_review",
            model_name="gpt-4o",
            prompt_tokens=1200,
            completion_tokens=400,
            cost_usd=0.012,
        )
    """

    async def record(
        self,
        session: AsyncSession,
        review_id: str,
        agent_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None = None,
    ) -> LLMUsageSchema:
        """
        Persist a single LLM usage record.

        If cost_usd is not provided, it is computed from the model's rate card.
        """
        if cost_usd is None:
            cost_usd = self._calculate_cost(model_name, prompt_tokens, completion_tokens)

        usage = LLMUsageSchema(
            review_id=review_id,
            agent_name=agent_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
        )

        repo = LLMUsageRepo(session)
        await repo.insert(usage)

        logger.info(
            "LLM usage recorded",
            review_id=review_id,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=f"${cost_usd:.4f}",
        )
        return usage

    def _calculate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Calculate cost in USD from the model rate card."""
        rates = MODEL_COST_MAP.get(model, {"prompt": 5.00, "completion": 15.00})
        return (
            prompt_tokens * rates["prompt"] / 1_000_000
            + completion_tokens * rates["completion"] / 1_000_000
        )

    def estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Public estimate method (no DB write) — useful for pre-call budgeting."""
        return self._calculate_cost(model, prompt_tokens, completion_tokens)