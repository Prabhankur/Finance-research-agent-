"""
nodes/summarizer.py  –  Rolling conversation summarization.

Runs once per turn, right after intent_router, before the turn branches to
chat_agent/smalltalk/report_handoff. Cheap on most turns: it's a token-count
check that returns instantly, and only calls an LLM when the conversation has
actually grown past the "recent, verbatim" window chat_agent_node keeps.

WHY a real rolling summary, not just a bigger window:
_window_messages() in nodes/chat_agent.py already drops old messages when
building the prompt for a single LLM call — but that's a READ-time trim, not
a WRITE-time one. The full, untrimmed history still lives forever in the
LangGraph checkpoint (growing the SQLite file), and the agent has no memory
at all of anything older than the window — ask about a company from 40
messages ago and it's simply gone, with no indication that it ever happened.

This node closes that gap: when the full stored history exceeds the keep-
window, everything OLDER than the window gets folded into a short LLM-written
summary (merged with any prior summary) and PERMANENTLY REMOVED from state via
LangGraph's RemoveMessage — not trimmed for one prompt, actually deleted from
the checkpoint. The summary travels forward in ChatState.conversation_summary
and is prepended to chat_agent's system prompt, so old context degrades
gracefully into a compressed fact instead of vanishing outright.

Self-limiting by construction: right after a fold, the old portion of history
is near-empty, so CHAT_SUMMARIZE_MIN_MESSAGES naturally prevents firing again
until enough NEW messages accumulate — no separate cooldown timer needed.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from config import CHAT_SUMMARIZE_MIN_MESSAGES, CHAT_SUMMARY_MAX_TOKENS
from graph.chat_state import ChatState
from nodes.chat_agent import _window_messages  # reuses the exact same pairing-safe boundary
from utils.groq_client import get_llm, invoke_with_retry

_SUMMARY_PROMPT = """You maintain a running summary of an ongoing financial-
research conversation, for a future turn that will NOT see the raw messages
below — only this summary.

Write a compact summary (under 200 words) covering: which companies/tickers
were discussed, which specific figures or metrics were already established
(WITH their values — a future turn should not have to re-fetch them), and
any question the user was in the middle of. Skip pleasantries and tool
mechanics. Plain prose, third person, no headers, no bullet lists.

If an EXISTING SUMMARY is given, merge the new messages into it — rewrite as
one coherent, still-compact summary rather than appending a second block."""


def _render_for_summary(messages: list) -> str:
    """
    Flattens messages into plain text for the summarizer's own prompt.

    Tool output is included (that's where the actual numbers live) but
    capped hard per message — this text is an INPUT to compression, not the
    thing being shown to a user, so there's no reason to let one huge filing
    dominate the summarization call's own token budget.
    """
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and str(m.content or "").strip():
            lines.append(f"Assistant: {m.content}")
        elif isinstance(m, ToolMessage):
            content = str(m.content)[:800]
            lines.append(f"[{m.name} result]: {content}")
    return "\n".join(lines)


def summarizer_node(state: ChatState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {}

    keep_tail = _window_messages(messages)
    fold_head = messages[: len(messages) - len(keep_tail)]

    if len(fold_head) < CHAT_SUMMARIZE_MIN_MESSAGES:
        return {}  # not worth an LLM call yet — see module docstring

    existing_summary = state.get("conversation_summary") or ""
    rendered = _render_for_summary(fold_head)

    if not rendered.strip():
        # Nothing worth summarizing (e.g. all tool-call scaffolding with no
        # renderable content) — still safe to drop these messages from state,
        # just skip the LLM call entirely.
        return {"messages": [RemoveMessage(id=m.id) for m in fold_head if getattr(m, "id", None)]}

    user_content = rendered
    if existing_summary:
        user_content = f"EXISTING SUMMARY:\n{existing_summary}\n\nNEW MESSAGES:\n{rendered}"

    try:
        llm = get_llm(temperature=0.0, max_tokens=CHAT_SUMMARY_MAX_TOKENS)
        response = invoke_with_retry(
            llm, [SystemMessage(content=_SUMMARY_PROMPT), HumanMessage(content=user_content)]
        )
        new_summary = str(response.content).strip()
    except Exception as exc:
        # Summarization failing must NEVER break the actual conversation —
        # worst case, history just isn't trimmed this turn and the check
        # runs again next turn once it's grown a bit further.
        print(f"[Summarizer] Failed ({exc}) — leaving history untrimmed this turn.")
        return {}

    removable_ids = [m.id for m in fold_head if getattr(m, "id", None)]
    print(
        f"[Summarizer] Folded {len(fold_head)} messages "
        f"({len(removable_ids)} removable) into a {len(new_summary)}-char summary."
    )

    return {
        "conversation_summary": new_summary,
        "messages": [RemoveMessage(id=mid) for mid in removable_ids],
    }
