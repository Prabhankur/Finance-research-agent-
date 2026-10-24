"""
graph/financial_agent_graph.py  –  The complete LangGraph state machine.

Architecture:

    query_elaboration -> roadmap_planner -> fetch_executor -> compute_executor
        -> context_router -> report_generator -> report_saver -> END

Both executors are deterministic-first: fetch_executor calls fetch tools
directly using args already known from the plan (no LLM decision per step —
nothing to drift from). compute_executor forces tool_choice onto the exact
planned tool (currently just calculator) since ITS inputs genuinely aren't
known until fetch has run, so an LLM has to read the gathered context and
fill in real numbers — that's necessary agentic work, not redundant
re-deciding.

Either executor can detour through a single shared `replanner` node when a
tool fails for a SYSTEMIC reason (broken dependency, auth failure — not just
a bad argument, which is handled by local retry inside the executor without
ever reaching here). The replanner revises the remaining plan and hands
control back to whichever executor was running, via `replan_phase` in
state. This is bounded by config.GLOBAL_REPLAN_BUDGET, shared across both
executors — not a per-step budget — so a genuinely broken data source
degrades gracefully (tagged unavailable, run continues) instead of looping.

    fetch_executor ──needs_replan──> replanner ──phase=="fetch"──> fetch_executor
         │no                              │phase=="compute"
         ▼                                ▼
    compute_executor <──────────── (back into loop)
         │no
         ▼
    context_router -> report_generator -> report_saver -> END
"""

from __future__ import annotations
from langgraph.graph import StateGraph, END

from graph.state import AgentState
from nodes import (
    query_elaboration_node,
    report_type_classifier_node,
    roadmap_planner_node,
    fetch_executor_node,
    compute_executor_node,
    replanner_node,
    context_router_node,
    report_generator_node,
    report_saver_node,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Conditional edge routers
# ─────────────────────────────────────────────────────────────────────────────

def route_after_fetch(state: AgentState) -> str:
    return "replan" if state.get("needs_replan") else "compute"


def route_after_compute(state: AgentState) -> str:
    return "replan" if state.get("needs_replan") else "router"


def route_after_replan(state: AgentState) -> str:
    # replan_phase was set by whichever executor escalated, and the
    # replanner itself doesn't touch it — just reads it here to know which
    # executor to hand control back to.
    return "fetch" if state.get("replan_phase") == "fetch" else "compute"


# ─────────────────────────────────────────────────────────────────────────────
#  Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_financial_agent_graph():
    """Assembles and compiles the full LangGraph financial agent."""

    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("query_elaboration",     query_elaboration_node)
    graph.add_node("report_type_classifier", report_type_classifier_node)
    graph.add_node("roadmap_planner",       roadmap_planner_node)
    graph.add_node("fetch_executor",        fetch_executor_node)
    graph.add_node("compute_executor",      compute_executor_node)
    graph.add_node("replanner",             replanner_node)
    graph.add_node("context_router",        context_router_node)
    graph.add_node("report_generator",      report_generator_node)
    graph.add_node("report_saver",          report_saver_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("query_elaboration")

    # ── Linear edges ──────────────────────────────────────────────────────────
    graph.add_edge("query_elaboration",       "report_type_classifier")
    graph.add_edge("report_type_classifier",  "roadmap_planner")
    graph.add_edge("roadmap_planner",         "fetch_executor")

    # ── Conditional edges: replan detour, shared by both executors ───────────
    graph.add_conditional_edges(
        "fetch_executor",
        route_after_fetch,
        {"replan": "replanner", "compute": "compute_executor"},
    )
    graph.add_conditional_edges(
        "compute_executor",
        route_after_compute,
        {"replan": "replanner", "router": "context_router"},
    )
    graph.add_conditional_edges(
        "replanner",
        route_after_replan,
        {"fetch": "fetch_executor", "compute": "compute_executor"},
    )

    # ── Exit path: route -> generate -> save -> END ───────────────────────────
    graph.add_edge("context_router",   "report_generator")
    graph.add_edge("report_generator", "report_saver")
    graph.add_edge("report_saver",     END)

    compiled = graph.compile()
    print("[Graph] Financial agent graph compiled successfully.")
    return compiled


# Module-level compiled graph
financial_agent = build_financial_agent_graph()
