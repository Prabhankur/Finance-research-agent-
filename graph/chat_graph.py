"""
graph/chat_graph.py  –  The conversational agent graph.

                    ┌──────────────┐
                    │ intent_router│  (rules first, LLM only if ambiguous)
                    └──────┬───────┘
        ┌──────────────────┼───────────────┬──────────────┐
        │ agent            │ smalltalk     │ handoff      │ end
        ▼                  ▼               ▼              ▼
   ┌──────────┐       ┌─────────┐    ┌──────────┐       END
   │chat_agent│◄──┐   │smalltalk│    │ handoff  │   (report offer
   └────┬─────┘   │   └────┬────┘    └────┬─────┘    already sent —
        │tools    │        ▼              ▼          waiting for a
        ▼         │       END            END         yes/no)
   ┌──────────┐   │
   │chat_tools├───┘  bounded by CHAT_MAX_TOOL_LOOPS
   └──────────┘

Memory: compiled with a checkpointer keyed on `thread_id`, so the caller
passes {"configurable": {"thread_id": "..."}} and LangGraph reloads the full
message history for that conversation automatically. The caller only ever
sends the NEW user message — never the whole transcript.

Checkpointer choice: MemorySaver (process RAM) by default, because it has zero
setup and this runs as a single API process. It's genuinely lost on restart.
`build_chat_graph(checkpointer=...)` takes an override, so swapping in
SqliteSaver or PostgresSaver for real persistence is a one-line change at the
call site with no graph edits — which is the point of injecting it rather than
hardcoding it inside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from graph.chat_state import ChatState
from nodes.chat_agent import (
    chat_agent_node,
    chat_tool_node,
    report_handoff_node,
    smalltalk_node,
)
from nodes.intent_router import intent_router_node, route_after_intent
from nodes.summarizer import summarizer_node
from config import CHAT_MAX_TOOL_LOOPS


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpointer
# ─────────────────────────────────────────────────────────────────────────────

# Where the SQLite checkpoint DB lives. A separate file from any future
# Postgres knowledge-layer database on purpose: this stores LangGraph's own
# conversation state (messages, intent, tool traces) — infrastructure for the
# chat mechanism itself — not financial facts. Different lifecycle, different
# concerns, no reason to couple them just because they'll eventually both be
# SQL. If the Postgres work later wants conversation state too, this file's
# contents migrate independently of the financial schema.
_CHECKPOINT_DB_PATH = Path(__file__).resolve().parent.parent / "output" / "chat_checkpoints.db"


def _default_checkpointer():
    """
    Returns a SQLite-backed checkpointer so conversations survive a server
    restart and can be listed (unlike MemorySaver, which has neither
    property — it's process RAM and offers no way to enumerate threads).

    This was originally MemorySaver, changed after being asked directly:
    "thread list" as a feature needs a checkpointer that supports listing
    multiple threads, and MemorySaver categorically cannot do that. SqliteSaver
    is the smallest change that unblocks it — same LangGraph checkpointer
    interface, just backed by a file instead of a dict, and it gets swapped
    for PostgresSaver later with no graph-level changes when the Postgres
    knowledge layer lands (same interface both times).

    Falls back to in-memory (with a loud warning) if the sqlite checkpoint
    package isn't installed, so a missing optional dependency degrades the
    dev experience instead of crashing the whole chat agent at import time.
    """
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        _CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_CHECKPOINT_DB_PATH), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()  # creates the checkpoint tables if they don't exist yet
        print(f"[ChatGraph] Using SqliteSaver at {_CHECKPOINT_DB_PATH}")
        return saver
    except ImportError:
        print(
            "[ChatGraph] WARNING: langgraph-checkpoint-sqlite not installed — "
            "falling back to in-memory checkpointing. Conversations will NOT "
            "survive a restart and thread listing will not work. "
            "Run: pip install langgraph-checkpoint-sqlite"
        )
        try:
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        except ImportError:
            from langgraph.checkpoint.memory import InMemorySaver  # type: ignore

            return InMemorySaver()


# ─────────────────────────────────────────────────────────────────────────────
#  Conditional edge
# ─────────────────────────────────────────────────────────────────────────────

def route_after_agent(state: ChatState) -> str:
    """
    After the agent speaks: run the tools it asked for, or end the turn.

    The loop-budget check is duplicated here even though chat_agent_node
    already unbinds tools when the budget is spent. Belt and braces: if a
    future edit to the agent node regresses that, this edge still stops the
    loop instead of letting it run until LangGraph's recursion limit throws an
    opaque error at the user.
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return "end"

    if state.get("tool_loops", 0) >= CHAT_MAX_TOOL_LOOPS:
        print("[ChatGraph] Tool loop budget hit at the edge — ending turn.")
        return "end"

    return "tools"


# ─────────────────────────────────────────────────────────────────────────────
#  Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_graph(checkpointer: Optional[Any] = None):
    """Assembles and compiles the conversational agent graph."""
    graph = StateGraph(ChatState)

    graph.add_node("intent_router", intent_router_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("chat_agent", chat_agent_node)
    graph.add_node("chat_tools", chat_tool_node)
    graph.add_node("smalltalk", smalltalk_node)
    graph.add_node("report_handoff", report_handoff_node)

    graph.set_entry_point("intent_router")

    # summarizer runs unconditionally after routing, before the turn branches.
    # It's a near-free token-count check on most turns (see nodes/summarizer.py
    # for why it's self-limiting) — cheaper to always run it here than to
    # duplicate the "is history long enough" check into every branch.
    graph.add_edge("intent_router", "summarizer")

    graph.add_conditional_edges(
        "summarizer",
        route_after_intent,
        {
            "agent": "chat_agent",
            "smalltalk": "smalltalk",
            "handoff": "report_handoff",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "chat_agent",
        route_after_agent,
        {"tools": "chat_tools", "end": END},
    )

    # Tools always hand control back to the agent to interpret their output.
    graph.add_edge("chat_tools", "chat_agent")
    graph.add_edge("smalltalk", END)
    graph.add_edge("report_handoff", END)

    compiled = graph.compile(checkpointer=checkpointer or _default_checkpointer())
    print("[ChatGraph] Chat agent graph compiled successfully.")
    return compiled


# Module-level compiled instance, mirroring financial_agent_graph.py's pattern.
chat_agent_graph = build_chat_graph()
