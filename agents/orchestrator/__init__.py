"""
agents/orchestrator/__init__.py

LangGraph Orchestrator — defines the full review pipeline as a compiled graph.

Graph topology:
    START
      │
    [clone_and_diff]         — clone repo, extract diff
      │
    [fan_out] ──────────────────────────────────────────┐
      │                      │                          │
    [static_analysis]  [security_analysis]        [llm_review]
      │                      │                          │
    [fan_in] ───────────────────────────────────────────┘
      │
    [merge_results]          — deduplicate, rank, score
      │
    [post_to_github]         — post review comments
      │
    [persist_results]        — save to PostgreSQL
      │
    [maybe_autofix]          — generate patches for HIGH/CRITICAL
      │
    END

All three analysis nodes execute concurrently via asyncio.gather().
LangGraph's Send API is used to fan out and merge state.

Architecture role:
  - Imported by the Worker — one pipeline instance per job
  - PipelineState is the LangGraph state object passed between nodes
  - Each node is a pure async function: state → state update dict
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.merger import MergerAgent
from agents.review import LLMReviewAgent
from agents.security import SecurityAnalysisAgent
from agents.static_analysis import StaticAnalysisAgent
from github.clone_repo import RepoCloner
from github.diff_extractor import DiffExtractor
from github.github_client import GitHubClient
from shared.constants import ReviewStatus, Severity, SeverityScore
from shared.logger import bind_request_context, get_logger
from shared.schemas import (
    LLMReviewResult,
    MergedReviewResult,
    PipelineState,
    SecurityAnalysisResult,
    StaticAnalysisResult,
)
from storage.db import get_db_session
from storage.repositories import (
    FindingRepo,
    LLMUsageRepo,
    PullRequestRepo,
    RepositoryRepo,
    ReviewRepo,
)
from tracing.cost_tracker import CostTracker

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Node Functions
# ------------------------------------------------------------------ #


async def node_clone_and_diff(state: PipelineState) -> dict[str, Any]:
    """
    Node 1: Clone the repository and extract the PR diff.

    Updates state with: diff, status=CLONING
    """
    logger.info("Node: clone_and_diff", review_id=state.review_id)
    payload = state.job.payload

    try:
        cloner = RepoCloner()
        local_path = await cloner.clone_and_checkout(payload)

        extractor = DiffExtractor()
        diff = extractor.extract(
            local_repo_path=local_path,
            repo_full_name=payload.repo_full_name,
            pr_number=payload.pr_number,
            head_sha=payload.pr_head_sha,
            base_sha=payload.pr_base_sha,
            base_branch=payload.repo_default_branch,
        )
        return {"diff": diff, "status": ReviewStatus.ANALYZING}

    except Exception as exc:
        logger.error("clone_and_diff node failed", error=str(exc))
        return {
            "errors": state.errors + [f"clone_and_diff: {str(exc)}"],
            "status": ReviewStatus.FAILED,
        }


async def node_concurrent_analysis(state: PipelineState) -> dict[str, Any]:
    """
    Node 2: Run Static, Security, and LLM Review agents concurrently.

    Uses asyncio.gather() for true parallel execution.
    Failures in individual agents are captured, not propagated.
    """
    if state.diff is None:
        logger.error("No diff available for analysis")
        return {"status": ReviewStatus.FAILED, "errors": state.errors + ["No diff"]}

    logger.info(
        "Node: concurrent_analysis",
        review_id=state.review_id,
        files=len(state.diff.changed_files),
    )

    static_agent = StaticAnalysisAgent()
    security_agent = SecurityAnalysisAgent()
    llm_agent = LLMReviewAgent()

    static_task = static_agent.analyze(state.diff)
    security_task = security_agent.analyze(state.diff)
    llm_task = llm_agent.review(state.diff)

    results = await asyncio.gather(
        static_task,
        security_task,
        llm_task,
        return_exceptions=True,
    )

    static_result, security_result, llm_result = results

    # Wrap exceptions in empty results so merger can still run
    if isinstance(static_result, Exception):
        logger.error("Static analysis failed", error=str(static_result))
        static_result = StaticAnalysisResult(
            errors=[str(static_result)]
        )

    if isinstance(security_result, Exception):
        logger.error("Security analysis failed", error=str(security_result))
        security_result = SecurityAnalysisResult(
            errors=[str(security_result)]
        )

    if isinstance(llm_result, Exception):
        logger.error("LLM review failed", error=str(llm_result))
        llm_result = LLMReviewResult(
            errors=[str(llm_result)]
        )

    return {
        "static_result": static_result,
        "security_result": security_result,
        "llm_result": llm_result,
    }


async def node_merge_results(state: PipelineState) -> dict[str, Any]:
    """
    Node 3: Merge all agent outputs into a single MergedReviewResult.
    """
    logger.info("Node: merge_results", review_id=state.review_id)

    merger = MergerAgent()
    merged = merger.merge(
        static_result=state.static_result or StaticAnalysisResult(),
        security_result=state.security_result or SecurityAnalysisResult(),
        llm_result=state.llm_result or LLMReviewResult(),
    )
    return {"merged_result": merged, "status": ReviewStatus.POSTING}


async def node_post_to_github(state: PipelineState) -> dict[str, Any]:
    """
    Node 4: Post review comments to GitHub.
    """
    logger.info("Node: post_to_github", review_id=state.review_id)

    if state.merged_result is None:
        return {"errors": state.errors + ["No merged result to post"]}

    payload = state.job.payload
    try:
        client = GitHubClient()
        await client.post_review(
            repo_full_name=payload.repo_full_name,
            pr_number=payload.pr_number,
            result=state.merged_result,
            head_sha=payload.pr_head_sha,
        )
    except Exception as exc:
        # Non-fatal: log and continue — we still want to persist results
        logger.error("Failed to post GitHub review", error=str(exc))
        return {"errors": state.errors + [f"post_to_github: {str(exc)}"]}

    return {}


async def node_persist_results(state: PipelineState) -> dict[str, Any]:
    """
    Node 5: Persist findings, review status, and LLM usage to PostgreSQL.
    """
    logger.info("Node: persist_results", review_id=state.review_id)

    payload = state.job.payload

    async with get_db_session() as session:
        repo_repo = RepositoryRepo(session)
        pr_repo = PullRequestRepo(session)
        review_repo = ReviewRepo(session)
        finding_repo = FindingRepo(session)
        llm_usage_repo = LLMUsageRepo(session)

        # Upsert repo and PR
        db_repo = await repo_repo.get_or_create(payload)
        db_pr = await pr_repo.get_or_create(payload, db_repo.id)

        # Ensure review record exists (created by worker before pipeline runs)
        # Update with completed results
        if state.merged_result:
            await review_repo.complete(state.review_id, state.merged_result)
            await finding_repo.bulk_insert(
                state.review_id, state.merged_result.findings
            )
        else:
            await review_repo.fail(state.review_id, "No merged result produced")

        # Track LLM usage cost
        if state.llm_result and state.llm_result.prompt_tokens > 0:
            tracker = CostTracker()
            await tracker.record(
                session=session,
                review_id=state.review_id,
                agent_name="llm_review",
                model_name=state.llm_result.model_used,
                prompt_tokens=state.llm_result.prompt_tokens,
                completion_tokens=state.llm_result.completion_tokens,
                cost_usd=state.llm_result.cost_usd,
            )

    return {"status": ReviewStatus.COMPLETED}


async def node_autofix(state: PipelineState) -> dict[str, Any]:
    """
    Node 6 (conditional): Generate auto-fix patches for HIGH/CRITICAL findings.

    Only runs if autofix is enabled in settings and high-severity findings exist.
    """
    from shared.config import get_settings
    settings = get_settings()

    if not settings.autofix_enabled:
        return {}

    if state.merged_result is None:
        return {}

    # Filter to findings that meet the autofix threshold
    from shared.constants import SeverityScore
    min_score = SeverityScore[settings.autofix_min_severity]
    eligible = [
        f for f in state.merged_result.findings
        if SeverityScore.from_severity(f.severity) >= min_score
    ]

    if not eligible:
        logger.info("No findings meet autofix threshold", threshold=settings.autofix_min_severity)
        return {}

    logger.info(
        "Node: autofix",
        review_id=state.review_id,
        eligible_findings=len(eligible),
    )

    from agents.autofix import AutoFixAgent
    from storage.repositories import FixPatchRepo

    autofix_agent = AutoFixAgent()
    patches = []

    async with get_db_session() as session:
        patch_repo = FixPatchRepo(session)

        for finding in eligible[:5]:  # Cap at 5 auto-fixes per review
            try:
                patch = await autofix_agent.generate_fix(
                    finding=finding,
                    repo_path=state.diff.local_repo_path if state.diff else "",
                    repo_full_name=state.job.payload.repo_full_name,
                    pr_number=state.job.payload.pr_number,
                    base_branch=state.job.payload.repo_default_branch,
                )
                if patch:
                    patches.append(patch)
                    await patch_repo.create(
                        review_id=state.review_id,
                        finding_id=finding.id,
                        patch=patch,
                    )
            except Exception as exc:
                logger.error(
                    "Auto-fix generation failed",
                    finding_id=finding.id,
                    error=str(exc),
                )

    return {"patches": patches}


# ------------------------------------------------------------------ #
# Graph Builder
# ------------------------------------------------------------------ #


def build_review_graph() -> Any:
    """
    Compile the LangGraph review pipeline.

    Returns a compiled graph ready for .ainvoke() calls.

    Graph edges:
      START → clone_and_diff → concurrent_analysis → merge_results
            → post_to_github → persist_results → autofix → END
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("clone_and_diff", node_clone_and_diff)
    graph.add_node("concurrent_analysis", node_concurrent_analysis)
    graph.add_node("merge_results", node_merge_results)
    graph.add_node("post_to_github", node_post_to_github)
    graph.add_node("persist_results", node_persist_results)
    graph.add_node("autofix", node_autofix)

    # Define edges (sequential, with concurrent_analysis doing internal gather)
    graph.add_edge(START, "clone_and_diff")
    graph.add_edge("clone_and_diff", "concurrent_analysis")
    graph.add_edge("concurrent_analysis", "merge_results")
    graph.add_edge("merge_results", "post_to_github")
    graph.add_edge("post_to_github", "persist_results")
    graph.add_edge("persist_results", "autofix")
    graph.add_edge("autofix", END)

    return graph.compile()


# Singleton compiled graph — built once per process
_review_graph = None


def get_review_graph():
    """Return the singleton compiled review graph."""
    global _review_graph
    if _review_graph is None:
        _review_graph = build_review_graph()
    return _review_graph