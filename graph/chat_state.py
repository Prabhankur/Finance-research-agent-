"""
graph/chat_state.py  –  State for the conversational chat agent graph.

Deliberately a SEPARATE TypedDict from graph/state.py's AgentState.

Why not reuse AgentState: it carries ~20 report-pipeline fields (roadmap,
rag_context, report_outline, replan_phase, final_report...) that mean nothing
in a chat turn. Reusing it would mean every chat checkpoint serialises a
mostly-empty report state, and every node reading `state["roadmap"]` would
need a guard. Two states, one shared tool layer, is the cleaner split — the
tools and the Groq client are what's genuinely shared, not the state shape.

Persistence: the chat graph is compiled with a checkpointer keyed on
thread_id, so `messages` accumulates across turns automatically. Per-turn
fields (intent, tool_loops, tool_trace) are reset at the start of each turn
by the intent router — see _reset_turn() there.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def reset_or_append(left: list, right: Any) -> list:
    """
    Reducer that appends by default but supports an explicit reset.

    Returning `None` from a node clears the list; returning a list appends.
    Plain `operator.add` can't express "reset" at all — returning [] appends
    nothing and leaves prior contents intact — which matters for genuinely
    per-turn accumulators: without a reset, successful_tool_results would
    grow for the life of the thread, and a synthesis failure on turn 10
    would replay tool results fetched for unrelated questions on turns 1-9.
    """
    if right is None:
        return []
    return (left or []) + list(right)


class ChatState(TypedDict, total=False):
    # ── Conversation ─────────────────────────────────────────────────────────
    # add_messages reducer: appends new messages, and de-duplicates by id so a
    # node returning an already-seen message doesn't double it up.
    messages: Annotated[list[BaseMessage], add_messages]

    # The raw text of the current user turn. Pulled out of messages by the
    # intent router so downstream nodes don't have to re-scan the history.
    user_input: str

    # ── Routing (set fresh every turn by intent_router_node) ─────────────────
    intent: str                 # one of config.CHAT_INTENTS
    intent_source: str          # "rules" | "llm" | "fallback"

    # ── Report handoff (human-in-the-loop) ───────────────────────────────────
    # When the router decides the user is asking for a full document rather
    # than a chat answer, we do NOT silently launch a 5-minute pipeline. We
    # stage it here and ask for confirmation. The API layer only kicks off the
    # report pipeline when report_confirmed flips true.
    pending_report_query: Optional[str]   # query to hand to the report graph
    pending_report_type: Optional[str]    # classified archetype, for display
    report_confirmed: bool                # user said yes on this turn

    # ── Tool loop control ────────────────────────────────────────────────────
    tool_loops: int             # agent<->tool round trips used this turn

    # Structured record of every tool call this turn, for the UI's "what did
    # it actually do" panel and for eval logging later. operator.add so the
    # tool node can append without clobbering.
    tool_trace: Annotated[list[dict[str, Any]], reset_or_append]

    # ── Partial-result preservation ──────────────────────────────────────────
    # Full, UNTRIMMED results of every tool call that succeeded this turn,
    # kept separately from tool_trace (which is metadata only — durations,
    # status, char counts — deliberately without payloads so it stays cheap
    # to stream to the UI on every event).
    #
    # This exists so a failure in the FINAL synthesis LLM call can't destroy
    # data that real API calls already fetched against a finite daily quota.
    # Before this, a 400 or a rate limit at the last step returned a bare
    # "I hit an error" and silently threw away several successful fetches.
    # See utils/result_formatter.build_partial_response.
    successful_tool_results: Annotated[list[dict[str, Any]], reset_or_append]

    # One of utils.result_formatter's four states. Distinguishes "couldn't
    # fetch anything" from "fetched everything, couldn't write it up" —
    # different situations, different messages, and later, different retry
    # behaviour (synthesis is free to retry; retrieval is not).
    turn_status: str

    # ── Cross-source reconciliation ──────────────────────────────────────────
    # Set by chat_agent_node right before the synthesis call. Populated ONLY
    # when the contradiction detector (see utils/contradiction_detector.py)
    # finds a meaningful disagreement between what a tool just returned and
    # what other providers previously said about the same fact. Streamed to
    # the UI for visibility and used by the evaluator to score whether the
    # answer actually acknowledged the disagreement or hid it.
    contradictions_detected: Annotated[list[dict[str, Any]], reset_or_append]

    # ── Rolling memory ────────────────────────────────────────────────────────
    # Set only by nodes/summarizer.py, read only by chat_agent's system prompt.
    # Plain str with NO reducer: whoever writes it overwrites it wholesale,
    # and every node that doesn't touch this key leaves the prior value
    # untouched — exactly "last write wins", which is what a single-writer
    # field wants. See nodes/summarizer.py for why this exists: it's the
    # compressed memory of everything that's aged out of the raw message
    # window, so old context degrades gracefully instead of vanishing.
    conversation_summary: Optional[str]

    # ── Lightweight conversational memory ────────────────────────────────────
    # Sticky context so "and its margins?" resolves without re-stating the
    # company. Updated opportunistically whenever a ticker is seen.
    active_ticker: Optional[str]
