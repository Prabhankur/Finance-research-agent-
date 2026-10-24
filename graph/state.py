"""
state.py  –  Shared agent state for the Financial Report LangGraph pipeline.

Every node reads from and writes to this TypedDict.
LangGraph merges updates automatically via the `add_messages` reducer
for the messages field; all other fields are last-write-wins.
"""

from __future__ import annotations
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Conversation / LLM messages (auto-appended by add_messages) ──────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── User input ────────────────────────────────────────────────────────────
    raw_query: str          # original question exactly as the user typed it
    ticker: str             # extracted stock ticker, e.g. "AAPL"

    # ── Node 1 output: elaborated query ──────────────────────────────────────
    elaborated_query: str   # enriched, clarified version of raw_query

    # ── Report-type routing (set before roadmap_planner runs) ────────────────
    # Which of the supported report archetypes this run is producing. Set by
    # either a user pre-selection (Streamlit sidebar / --report-type flag) or
    # by the classifier node. Read by roadmap_planner (to shape the plan) and
    # report_generator (to pick the right SECTION_INSTRUCTIONS keys).
    # Valid values in config.REPORT_TYPES:
    #   COMPANY_RESEARCH, COMPARATIVE_ANALYSIS, EARNINGS_EVENT,
    #   VALUATION_RISK, MACRO_COUNTRY, OTHER
    report_type: str
    report_type_source: str   # "user" | "classifier" | "fallback" — shown in UI

    # ── Node 2 output: research roadmap ──────────────────────────────────────
    roadmap: dict           # {steps: [...], report_outline: {...}}

    # ── Tool-loop state ───────────────────────────────────────────────────────
    tool_calls_made: list[dict]   # history of every tool call attempt + outcome
    # Each entry: {step, tool, status: "success"|"failed"|"skipped",
    #              content, retries_used}. Replaces the old flat list[str] —
    #              a plain string couldn't distinguish real data from an
    #              error message, so failed calls used to look identical to
    #              successful ones by the time they reached the report.
    context_chunks: list[dict]
    iteration: int                # safety counter – prevents infinite loops

    # ── Failure recovery state ────────────────────────────────────────────────
    disabled_tools: list[str]     # tools blacklisted this run after a
                                   # systemic failure (e.g. dependency missing)
    replan_count: int             # how many times the replanner has run so
                                   # far this run, checked against the global
                                   # budget in config.GLOBAL_REPLAN_BUDGET

    # Transient routing signals set by an executor, read by the conditional
    # edge to decide whether to detour through the replanner before
    # continuing. Cleared (set back to False/None) once the replanner runs.
    needs_replan: bool
    replan_reason: Optional[dict]     # the failed context_chunks-style entry
    replan_phase: Optional[str]       # "fetch" or "compute" — which executor
                                        # to return control to after replanning

    # ── RAG / report inputs ──────────────────────────────────────────────────
    rag_context: str              # routed context for report writer, built
                                   # from context_chunks by the context router
    report_outline: dict          # section plan passed to report generator

    # ── Final output ─────────────────────────────────────────────────────────
    final_report: str             # full markdown report text
    report_path: str              # absolute path to saved .docx / .md file
