"""
nodes/chat_agent.py  –  The conversational agent: a BOUNDED ReAct loop.

    intent_router -> chat_agent -> (tool_calls?) -> chat_tools -> chat_agent -> ...
                                        no
                                        v
                                       END

How this differs from the report pipeline, and why both exist:

    report pipeline          chat agent
    ---------------          ----------
    plan everything up       decide one step at a time
    front (roadmap)
    optimises for            optimises for latency (a user is waiting)
    completeness
    ~15 tool calls           <= CHAT_MAX_TOOL_LOOPS round trips
    context router shards    tool output trimmed inline
    a huge context
    output: .md + .docx      output: one chat message

They are not competing designs — they're the right shape for different jobs.
Planning ahead is correct when you're producing a 25-section document and can
amortise the planning cost. Reacting step-by-step is correct when the user
asked one question and is watching a spinner.

Three failure modes this node is specifically built to survive:

1. Runaway loops. The agent gets CHAT_MAX_TOOL_LOOPS round trips. On the last
   one it is re-invoked with NO tools bound, which makes answering the only
   available action. It can't loop, and it can't return an empty turn.

2. Context blowup. A 10-K is ~500k characters. Pasted raw into the message
   history it exceeds the context window on a single call. Every tool result
   is trimmed to CHAT_TOOL_RESULT_MAX_CHARS before it re-enters the
   conversation, and the history itself is windowed.

3. Orphaned tool messages. If history trimming cuts between an AIMessage
   carrying tool_calls and its matching ToolMessages, the provider rejects the
   whole request with a confusing schema error. _window_messages() is written
   to never cut inside that pair.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import date
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from config import (
    CHAT_ENABLED_TOOLS,
    CHAT_HISTORY_MAX_MESSAGES,
    CHAT_MAX_CONTEXT_TOKENS,
    CHAT_MAX_PARALLEL_TOOL_CALLS,
    CHAT_MAX_TOOL_LOOPS,
    CHAT_SERIAL_TOOLS,
    CHAT_TOOL_RESULT_MAX_CHARS,
)
from graph.chat_state import ChatState
from config import CHAT_MAX_BATCH_TICKERS, INTENT_OUT_OF_SCOPE, LOCAL_RETRY_LIMIT, TRANSIENT_RETRY_PAUSE_S
from utils.text_budget import fair_share
from tools import TOOLS_BY_NAME as _SHARED_TOOLS_BY_NAME
from tools.chart_tool import CHART_TOOL_NAME, render_chart
from utils.error_classifier import LEGITIMATE_EMPTY, OK, SYSTEMIC, TRANSIENT, BAD_ARGS, classify_result
from utils.groq_client import get_llm, invoke_with_retry
from utils.contradiction_detector import (
    find_contradictions_from_state,
    format_contradictions_note,
)
from utils.result_formatter import (
    COMPLETE,
    RETRIEVAL_COMPLETE_SYNTHESIS_FAILED,
    RETRIEVAL_FAILED,
    build_partial_response,
)

# Chat's own tool registry = the shared report-pipeline tools + render_chart.
# Kept separate from tools/__init__.py's ALL_TOOLS (see CHAT_ENABLED_TOOLS in
# config.py for why render_chart doesn't belong in the shared list). Rebuilt
# as a plain dict merge rather than mutating the imported one, so nothing
# chat does here can leak render_chart into the report pipeline's tool set.
TOOLS_BY_NAME = {**_SHARED_TOOLS_BY_NAME, CHART_TOOL_NAME: render_chart}

# Only the tools chat is allowed to use, in config order.
CHAT_TOOLS = [TOOLS_BY_NAME[name] for name in CHAT_ENABLED_TOOLS if name in TOOLS_BY_NAME]


# ─────────────────────────────────────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────────────────────────────────────

def _system_prompt(
    active_ticker: str | None,
    tools_available: bool,
    conversation_summary: str | None = None,
    contradiction_note: str | None = None,
    *,
    active_tickers: list[str] | None = None,
    brief: dict | None = None,
) -> str:
    today = date.today().isoformat()

    base = f"""You are a financial research analyst assistant. Today is {today}.

You answer questions about companies, filings, earnings, valuation, and macro
data using real data tools — never from memory.

## Hard rules on numbers

- NEVER state a specific financial figure (revenue, EPS, P/E, margin, price,
  debt, growth rate) unless it came from a tool result THIS conversation or
  the summary below. If you don't have it, say so and offer to fetch. Never
  invent plausible-looking numbers.
- Use `calculator` for ALL arithmetic, never mental math. Plain arithmetic:
  metric sum/subtract/multiply/divide/average with a `values` list (four
  quarters: values=[q1,q2,q3,q4]); `percent_change` for a simple change;
  `cagr` ONLY for multi-year compound growth — never to add.
- CURRENCY: figures are in each company's own currency (see the data header,
  e.g. [reporting currency: INR]). Never add, compare or rank across different
  currencies — say they differ. Convert ONLY when the user explicitly asks:
  fx_rate, then calculator multiply. Non-US tickers need a suffix (.NS .BO .L .TO).
- NEVER sum per-capita values, rates, percentages, ratios or index values
  across periods — they are not additive. Quarterly LEVEL series (e.g.
  NA000334Q, millions) may be summed to a year; if an annual series exists
  (e.g. GDPA), fetch it instead.

## Source citation with evidence tiers

Cite every figure with its source tier:
- Tier A: SEC filings, FRED — regulatory/official
- Tier B: yfinance, Alpha Vantage — commercial providers, subject to
  computation choices (e.g. ROE using average vs ending equity)
- Tier C: earnings transcripts — management-provided
- Tier D: web search — unstructured, least reliable

Format: "(source, tier X)" e.g. "(yfinance, tier B)". When two sources
disagree, lead with the higher tier and disclose the delta rather than
silently averaging. Convert raw units — $391,035,000,000 → $391.0B.

## Tool usage

- For 2+ companies make ONE financial_data_api call with `tickers=[...]` (max 8),
  never one call per company — each extra round re-sends the whole prompt.
  Other independent calls (different tools) go in the same message.
- Don't re-fetch data already in context or the summary.
- Use `fields` on `financial_data_api` when you only need specific metrics —
  a comparison across 3 companies with fields=["revenue","operating_margin"]
  is roughly 5× smaller than the default full response. Do this whenever
  you know exactly what you need.
- `earnings_transcript` is rate-limited (25/day) — use only when the user
  asks about what management said on a call.
- Tool errors: a tool reported "unavailable" must not be called again — use
  another source or answer with what you have. Fix bad arguments once; never
  repeat an identical failing call. "No data exists" is a valid answer: say it
  is not available. Say what failed. Never guess.
- Use `render_chart` for comparisons/trends that read better as a chart.
  Only chart real numbers already fetched this turn.

## Style

- Chat, not a document. Usually 2-6 sentences, add a small table if there
  are several numbers. Lead with the answer, then supporting numbers, then
  caveats. No preamble ("Great question!") or sales pitch.
- Not a licensed advisor. For "should I buy" questions, give analysis and
  trade-offs, and be clear the decision and risk are the user's.
"""

    if contradiction_note:
        base += "\n" + contradiction_note + "\n"

    if conversation_summary:
        base += (
            f"\n## Earlier in this conversation (summarized)\n"
            f"{conversation_summary}\n\n"
            f"This is a COMPRESSED summary of older messages that are no "
            f"longer in the raw conversation below — treat facts in it as "
            f"already-established, exactly like a tool result, and cite them "
            f"the same way (e.g. \"as established earlier, ...\").\n"
        )

    if active_tickers and len(active_tickers) > 1:
        # Several companies are in focus (e.g. an 8-company comparison). The old
        # single-ticker sentence told the model the topic was the FIRST one.
        base += (
            f"\n## Conversation context\n"
            f"The companies under discussion are **{', '.join(active_tickers)}**. "
            f"A bare \"it\" or \"its\" is ambiguous with several companies: resolve it "
            f"only if the question makes it clear, otherwise say which one you assumed.\n"
        )
    elif active_ticker:
        base += (
            f"\n## Conversation context\n"
            f"The company currently under discussion is **{active_ticker}**. "
            f"Resolve bare pronouns and follow-ups (\"its margins\", \"what about "
            f"last year\") against this ticker unless the user clearly switches "
            f"to a different company.\n"
        )
    if brief:
        from nodes.query_understanding import brief_for_prompt
        base += brief_for_prompt(brief)

    if not tools_available:
        base += (
            "\n## IMPORTANT — final turn\n"
            "You have used your tool budget for this question. Answer NOW using "
            "only what has already been gathered above. State clearly which "
            "parts you could not verify. Do not claim you will fetch anything.\n"
        )

    return base


# ─────────────────────────────────────────────────────────────────────────────
#  Token estimation
# ─────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text))

except Exception:
    # tiktoken's BPE isn't identical to Llama's tokenizer (Groq's models), so
    # this was always an ESTIMATE, not an exact count, even with it installed.
    # If it's missing, fall back to a chars/4 approximation — a well-known
    # rough-and-ready ratio for English text — rather than hard-failing the
    # whole chat agent over an optional dependency.
    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def _message_tokens(message: BaseMessage) -> int:
    return _count_tokens(str(getattr(message, "content", "") or ""))


# ─────────────────────────────────────────────────────────────────────────────
#  History windowing
# ─────────────────────────────────────────────────────────────────────────────

def _window_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Keeps recent history under BOTH a message-count cap and a token-budget
    cap, without ever orphaning a ToolMessage from the AIMessage that
    requested it.

    Why both caps: message-count alone isn't a safe bound on request size. A
    3-way company comparison can put several ~5000-char tool results into a
    handful of messages, blowing the token budget while still comfortably
    under CHAT_HISTORY_MAX_MESSAGES — which is what happened in production
    (see config.py's CHAT_MAX_CONTEXT_TOKENS note). So this trims by message
    count first, then by token count on top, and the token pass is the one
    that actually protects against a provider 400 on oversized requests.

    Providers validate that every ToolMessage's tool_call_id matches a
    tool_call on a preceding AIMessage. A naive slice breaks that whenever a
    turn used tools, and the resulting 400 is opaque. So after EACH trimming
    pass we drop any leading ToolMessages (and their now-unsatisfied parent
    AIMessage) that lost their pairing.
    """
    def _drop_orphans(msgs: list[BaseMessage]) -> list[BaseMessage]:
        out = list(msgs)
        while out and isinstance(out[0], ToolMessage):
            out.pop(0)
        if out and isinstance(out[0], AIMessage) and getattr(out[0], "tool_calls", None):
            satisfied = {m.tool_call_id for m in out if isinstance(m, ToolMessage)}
            wanted = {tc["id"] for tc in out[0].tool_calls}
            if not wanted.issubset(satisfied):
                out.pop(0)
                while out and isinstance(out[0], ToolMessage):
                    out.pop(0)
        return out

    windowed = messages[-CHAT_HISTORY_MAX_MESSAGES:] if len(messages) > CHAT_HISTORY_MAX_MESSAGES else list(messages)
    windowed = _drop_orphans(windowed)

    # Token pass: drop oldest messages (in pairing-safe chunks) until under
    # budget. Walking forward and re-checking after each orphan-drop is
    # simpler and safer than trying to compute the exact cut point up front —
    # this runs on at most ~24 messages, so the extra passes are free.
    total = sum(_message_tokens(m) for m in windowed)
    while windowed and total > CHAT_MAX_CONTEXT_TOKENS:
        removed = windowed.pop(0)
        total -= _message_tokens(removed)
        windowed = _drop_orphans(windowed)
        # Recompute total after orphan-dropping may have removed more than one.
        total = sum(_message_tokens(m) for m in windowed)

    return windowed


# ─────────────────────────────────────────────────────────────────────────────
#  Agent node
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

_TOOL_FAILURE_SHAPE = _re.compile(
    r"tool_use_failed|Tool choice is none, but model called a tool",
    _re.IGNORECASE,
)


def _looks_like_tool_use_failure(exc: Exception) -> bool:
    """Narrow check — only trigger the rescue path for the specific failure
    shape this exists to fix. A 429 or a genuine 500 should NOT trigger an
    extra LLM call; those are handled by their own paths and a rescue
    attempt would just burn another call on a problem it can't fix."""
    return bool(_TOOL_FAILURE_SHAPE.search(str(exc)))


def _condense_for_rescue(preserved_results: list[dict], max_chars: int = 4000) -> str:
    """
    Renders saved tool results as plain text (NOT the original tool-call /
    tool-response message shape — that history is what the rescue exists to
    avoid replaying).

    The character budget is shared FAIRLY. The previous version re-divided
    whatever was left by the TOTAL count on every pass, so with 5 companies
    the first kept ~500 chars and the last ~205 — later companies were
    silently cut harder, mid-table. Now: a result shorter than its fair share
    keeps everything and the unused budget is handed to the longer ones, and
    anything cut carries an explicit [truncated] marker so the model knows the
    data is incomplete instead of treating a cut table as the full table.
    """
    blocks = [
        f"[{r.get('tool', '?')}] {r.get('args', {})}\n{str(r.get('result') or '')}"
        for r in preserved_results
    ]
    return "\n\n".join(fair_share(blocks, max_chars)) if blocks else ""


def _plain_history(messages: list, keep: int = 4, per_msg_chars: int = 600) -> str:
    """
    Earlier questions and final answers of this conversation as PLAIN TEXT.

    Without this the rescue only saw the current sentence, so a follow-up like
    "I asked for full year gdp so add them all" lost what "them" referred to.
    Tool calls and tool results are deliberately excluded — only messages a
    person would recognise as part of the conversation — so this cannot
    reintroduce the tool-call history the rescue is built to avoid.
    """
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=None
    )
    earlier = messages[:last_human] if last_human is not None else []
    lines = []
    for m in earlier:
        content = str(getattr(m, "content", "") or "").strip()
        if not content:
            continue
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            role = "Assistant"
        else:
            continue
        if len(content) > per_msg_chars:
            content = content[:per_msg_chars] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines[-keep:])


def _attempt_rescue_synthesis(
    user_question: str,
    preserved_results: list[dict],
    active_ticker: str | None,
    conversation_summary: str | None = None,
    contradiction_note: str | None = None,
    history_text: str = "",
    active_tickers: list[str] | None = None,
    brief: dict | None = None,
) -> str | None:
    """
    One last synthesis attempt with a FRESH, minimal message list — no tools
    declared, and no prior tool-call messages. Returns the answer text, or
    None if this also fails (caller then falls back to the raw partial
    response).

    It now uses the same _system_prompt() builder as the normal path, in its
    tools_available=False mode ("you have used your tool budget... answer NOW
    ... do not claim you will fetch anything"). That brings back what the
    hand-written rescue prompt dropped: the citation/tier rules, the rolling
    conversation summary, and the cross-source contradiction note. Previously
    a rescued answer could ignore a conflict the UI banner was showing.
    """
    system_text = _system_prompt(
        active_ticker,
        tools_available=False,
        conversation_summary=conversation_summary,
        contradiction_note=contradiction_note,
        active_tickers=active_tickers,
        brief=brief,
    )
    if active_tickers and len(active_tickers) > 1:
        ticker_note = f"\n\nCompanies in focus: {', '.join(active_tickers)}"
    else:
        ticker_note = f"\n\nCompany in focus: {active_ticker}" if active_ticker else ""
    history_block = (
        f"Earlier in this conversation (context only):\n{history_text}\n\n"
        if history_text else ""
    )
    user_content = (
        f"Question: {user_question}{ticker_note}\n\n"
        f"{history_block}"
        f"Retrieved data (already fetched this turn):\n"
        f"{_condense_for_rescue(preserved_results)}\n\n"
        "Write the answer now using only the retrieved data above. You cannot "
        "fetch or calculate anything further. Do not do multi-digit arithmetic "
        "yourself: if a derived figure was not already computed above, give its "
        "inputs and say it was not calculated. If a needed figure is missing, "
        "say so plainly. Do not mention tools or errors."
    )

    try:
        # Deliberately NOT using get_llm's with_tools/tool_choice: a request
        # with zero tools declared has nothing for Groq's validator to reject
        # a tool call against. max_tokens is 2000 (was 1500) because on
        # reasoning models that budget also pays for the model's thinking; a
        # trace showed one call spend 1,361 tokens on reasoning alone, which
        # would leave 1500 almost nothing for the visible answer.
        llm = get_llm(temperature=0.2, max_tokens=2000)
        response = invoke_with_retry(
            llm,
            [SystemMessage(content=system_text), HumanMessage(content=user_content)],
        )
        text = str(response.content or "").strip()
        return text if text else None
    except Exception as exc:
        print(f"[ChatAgent] Rescue synthesis attempt failed: {exc}")
        return None


def chat_agent_node(state: ChatState) -> dict:
    """One reasoning step: either emit tool calls, or emit the final answer."""
    loops = state.get("tool_loops", 0)
    # Tools withdrawn this turn (systemic failure) are not even bound: the model
    # cannot pick a tool it is not shown, and the schemas it no longer carries
    # save tokens. _run_one still refuses them if it hallucinates one anyway.
    disabled = set(state.get("disabled_tools") or [])
    active_tools = [t for t in CHAT_TOOLS if t.name not in disabled]
    budget_left = loops < CHAT_MAX_TOOL_LOOPS and bool(active_tools)

    history = _window_messages(state.get("messages", []))

    # Contradiction detection runs BEFORE the synthesis call — that's the
    # entire point, otherwise the model has already committed to whichever
    # number it saw first. Fully deterministic, DB-driven, returns [] on
    # any failure so it can never break the turn. See
    # utils/contradiction_detector.py for the rules and the tier system.
    contradictions = find_contradictions_from_state(dict(state))
    contradiction_note = format_contradictions_note(contradictions) if contradictions else None
    if contradictions:
        print(
            f"[ChatAgent] {len(contradictions)} contradiction(s) detected — "
            f"injecting reconciliation note into synthesis prompt."
        )
    prompt = _system_prompt(
        state.get("active_ticker"),
        tools_available=budget_left,
        conversation_summary=state.get("conversation_summary"),
        contradiction_note=contradiction_note,
        active_tickers=state.get("active_tickers"),
        brief=state.get("brief"),
    )
    messages = [SystemMessage(content=prompt)] + history

    try:
        if budget_left:
            llm = get_llm(temperature=0.2, max_tokens=2048, with_tools=active_tools)
        else:
            # Final round. Tools stay BOUND but tool_choice is explicitly
            # "none" — see utils/groq_client._FallbackChatGroq._build for why
            # this is stronger than simply omitting the tools list (short
            # version: gpt-oss models emitted tool calls anyway when tools
            # were merely absent, and Groq 400'd the entire request).
            print(f"[ChatAgent] Tool budget exhausted ({loops}) — forcing final answer.")
            llm = get_llm(
                temperature=0.2, max_tokens=2048,
                with_tools=active_tools or CHAT_TOOLS, tool_choice="none",
            )

        response = invoke_with_retry(llm, messages)
    except Exception as exc:
        print(f"[ChatAgent] LLM call failed: {exc}")

        preserved = state.get("successful_tool_results", []) or []
        failed_traces = [
            t for t in (state.get("tool_trace", []) or []) if t.get("status") == "failed"
        ]

        # RESCUE ATTEMPT: one more synthesis call, but with a FRESH message
        # list instead of replaying the tool-call-laden history that likely
        # caused the failure in the first place.
        #
        # Real evidence this matters: groq_client._FallbackChatGroq already
        # retries a tool_use_failed error once, stripping the `tools` field
        # — but it replays the SAME messages, which still contain prior
        # AIMessages with tool_calls attached from earlier this turn.
        # Production traces showed the retry failing with the IDENTICAL
        # error text, which is consistent with the model treating its own
        # conversation history — not the current request's tools field — as
        # license to keep emitting tool-call-shaped output. This rescue
        # attempt sidesteps that by not showing the model any tool-call
        # history at all: just the question and a condensed rendering of
        # what was actually retrieved, phrased as a request to write the
        # answer now.
        if preserved and _looks_like_tool_use_failure(exc):
            print(
                f"[ChatAgent] Synthesis failed with a tool-call-shaped error. "
                f"Attempting rescue synthesis with a condensed, tool-history-"
                f"free prompt ({len(preserved)} preserved result(s))..."
            )
            rescued = _attempt_rescue_synthesis(
                user_question=state.get("user_input", ""),
                preserved_results=preserved,
                active_ticker=state.get("active_ticker"),
                conversation_summary=state.get("conversation_summary"),
                contradiction_note=contradiction_note,
                history_text=_plain_history(state.get("messages", [])),
                active_tickers=state.get("active_tickers"),
                brief=state.get("brief"),
            )
            if rescued:
                print("[ChatAgent] Rescue synthesis succeeded.")
                return {
                    "messages": [AIMessage(content=rescued)],
                    "turn_status": COMPLETE,
                    "contradictions_detected": contradictions if contradictions else None,
                    # Tagged so evaluation can tell a rescued answer from a
                    # normal one — both are COMPLETE, but the rescue ran with
                    # narrower context and should be measured separately.
                    "synthesis_mode": "rescue",
                }
            print("[ChatAgent] Rescue synthesis also failed — falling back to partial response.")

        # THE IMPORTANT CASE: real API calls already succeeded this turn.
        # Their data cost real quota and must not be discarded just because
        # the write-up step failed. Rendered deterministically in Python,
        # which cannot fail the same way the LLM call just did.
        if preserved:
            print(
                f"[ChatAgent] Synthesis failed but {len(preserved)} tool "
                f"result(s) preserved — returning partial response."
            )
            return {
                "messages": [
                    AIMessage(content=build_partial_response(preserved, exc, failed_traces))
                ],
                "turn_status": RETRIEVAL_COMPLETE_SYNTHESIS_FAILED,
            }

        # Was: showing only `type(exc).__name__` to the user, so every
        # failure looked like the same opaque "BadRequestError" with no way
        # to tell a context-length overflow apart from a malformed request
        # apart from a genuine outage — the actual detail only ever reached
        # the server terminal. A truncated, best-effort-sanitized slice of
        # the real message is far more actionable, and short enough that it
        # shouldn't leak anything sensitive (Groq error bodies describe the
        # REQUEST shape, not secrets — no API key ever appears in exc text).
        detail = str(exc).strip().replace("\n", " ")
        if len(detail) > 300:
            detail = detail[:300] + "…"

        hint = ""
        lowered = detail.lower()
        if "daily token quota" in lowered or "per day" in lowered:
            # Checked BEFORE the generic rate-limit branch below on purpose:
            # this message (from groq_client.invoke_with_retry) already says
            # retrying won't help — appending "wait a few seconds" under it
            # would directly contradict that and re-encourage exactly the
            # wasted retry loop this error class exists to prevent.
            hint = ""  # message is already self-contained and specific
        elif "context" in lowered or "too long" in lowered or "maximum" in lowered:
            hint = " This usually means the conversation got too long for one request — try `/new` or ask a narrower question."
        elif "rate" in lowered or "429" in detail:
            hint = " This is the Groq rate limit — wait a few seconds and try again."

        return {
            "messages": [
                AIMessage(
                    content=(
                        f"I hit an error talking to the language model: "
                        f"`{type(exc).__name__}: {detail}`.{hint}"
                    )
                )
            ],
            "turn_status": RETRIEVAL_FAILED,
        }

    tool_calls = getattr(response, "tool_calls", None) or []

    # Guard the fan-out width. A model that asks for 12 tools in one step will
    # blow the AV daily limit in a single turn.
    if len(tool_calls) > CHAT_MAX_PARALLEL_TOOL_CALLS:
        print(
            f"[ChatAgent] Trimming {len(tool_calls)} tool calls -> "
            f"{CHAT_MAX_PARALLEL_TOOL_CALLS}"
        )
        response.tool_calls = tool_calls[:CHAT_MAX_PARALLEL_TOOL_CALLS]
        tool_calls = response.tool_calls

    if tool_calls:
        names = ", ".join(tc["name"] for tc in tool_calls)
        print(f"[ChatAgent] Round {loops + 1}: calling {names}")
        return {"messages": [response]}

    print(f"[ChatAgent] Final answer produced after {loops} tool round(s).")
    return {
        "messages": [response],
        "turn_status": COMPLETE,
        "contradictions_detected": contradictions if contradictions else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Tool node
# ─────────────────────────────────────────────────────────────────────────────

def _call_key(name: str, args: dict) -> str:
    """Stable identity of a tool call (name + arguments) for repeat detection."""
    return f"{name}|{json.dumps(args, sort_keys=True, default=str)}"


def _run_one(
    tool_call: dict,
    disabled: frozenset = frozenset(),
    failed_calls: frozenset = frozenset(),
) -> dict:
    """
    Executes ONE tool call under the failure policy, and returns a trace record.

    Policy, by failure type (see utils/error_classifier.py):
      TRANSIENT (timeout, dropped connection, 5xx) -> retried ONCE here, after a
          short pause, with no model involved. A blip needs no reasoning, and
          routing it through the model costs a whole round of the 4-round budget
          plus an LLM call (~5-6k prompt tokens in production traces).
      SYSTEMIC (bad/missing key, persistent rate limit) -> never retried. The
          caller withdraws the tool for the rest of the turn; if the model asks
          for it anyway it is refused here without executing.
      BAD_ARGS -> NOT retried here: the arguments must change, and only the
          model can change them. It is told what went wrong. An identical repeat
          of a call that already failed this way is refused (same args, same
          error — it can only waste a round).
      LEGITIMATE_EMPTY -> returned with an explicit "stop, this is the answer".
    """
    name = tool_call["name"]
    args = tool_call.get("args", {}) or {}
    started = time.time()

    def record(status: str, content: str, error_class: str, retries: int = 0) -> dict:
        return {
            "tool": name, "args": args, "status": status,
            "error_class": error_class, "content": content,
            "duration_s": round(time.time() - started, 2),
            "tool_call_id": tool_call["id"], "retries": retries,
        }

    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return record("failed", f"Unknown tool '{name}'. Available: {', '.join(TOOLS_BY_NAME)}", BAD_ARGS)

    if name in disabled:
        return record(
            "failed",
            f"[{name}] Unavailable for the rest of this turn: it failed earlier with an "
            f"error that retrying cannot fix (e.g. missing/invalid API key or a rate "
            f"limit). Do not call it again — use a different source or answer with "
            f"what you already have.",
            SYSTEMIC,
        )
    if _call_key(name, args) in failed_calls:
        return record(
            "failed",
            f"[{name}] This exact call already failed earlier this turn because of its "
            f"arguments. Do not repeat it unchanged — correct the arguments or take a "
            f"different approach.",
            BAD_ARGS,
        )

    retries = 0
    category = OK
    raw = ""
    for attempt in range(LOCAL_RETRY_LIMIT + 1):
        try:
            raw = str(tool.invoke(args))
        except Exception as exc:
            # Classified from its text like any other failure. This used to be
            # hard-wired to "systemic", so a plain TypeError (bad arguments)
            # or a dropped connection was treated as a permanently broken tool.
            raw = f"[{name}] Error: raised {type(exc).__name__}: {exc}"
        category = classify_result(name, raw)
        if category == TRANSIENT and attempt < LOCAL_RETRY_LIMIT:
            retries += 1
            time.sleep(TRANSIENT_RETRY_PAUSE_S)
            continue
        break

    # The classification is reused from the report pipeline so "no filings
    # found" and "auth failed" stay distinguishable in chat too. An empty result
    # means stop asking; a failure means the model should try a different route.
    status = {OK: "success", LEGITIMATE_EMPTY: "empty"}.get(category, "failed")
    content = raw
    if status == "empty":
        if not raw.strip():
            content = f"[{name}] No data available for these arguments."
        content += ("\n[No data exists for these arguments — this is a valid result. "
                    "Do not retry; tell the user it is not available.]")
    return record(status, content, category, retries)


def _multi_token(value) -> bool:
    """True for a `ticker` string that actually carries several tickers."""
    return isinstance(value, str) and bool(_re.search(r"[,;\s]", value.strip()))


def _expand_batch_calls(tool_calls: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """
    Turns financial_data_api(tickers=[A, B, C]) into one internal call per company.

    Why here and not inside the tool: each company then runs through _run_one like
    any other tool call — its own retry-on-transient, withdraw-on-systemic, trace
    row and preserved result — instead of being buried inside one opaque call.
    The model still sees ONE call and gets ONE ToolMessage back (providers require
    exactly one reply per tool_call_id); see the merge in chat_tool_node.

    Returns (internal_calls, groups) where groups maps an original call id to
    {"ids": [internal ids], "skipped": [tickers not fetched]}.
    """
    from tools import normalize_tickers

    internal: list[dict] = []
    groups: dict[str, dict] = {}
    budget = CHAT_MAX_BATCH_TICKERS          # companies fetched per round, all batch calls together
    for tc in tool_calls:
        args = dict(tc.get("args") or {})
        if tc["name"] != "financial_data_api" or not (args.get("tickers") or _multi_token(args.get("ticker"))):
            internal.append(tc)
            continue
        kept, skipped = normalize_tickers(args.get("ticker"), args.get("tickers"), limit=max(budget, 0))
        if not kept:
            internal.append(tc)              # let the tool answer with its own clear error
            continue
        budget -= len(kept)
        base = {k: v for k, v in args.items() if k not in ("ticker", "tickers")}
        if len(kept) == 1 and not skipped:
            internal.append({**tc, "args": {**base, "ticker": kept[0]}})
            continue
        ids = []
        for i, ticker in enumerate(kept):
            call_id = f"{tc['id']}::{i}"
            ids.append(call_id)
            internal.append({"name": tc["name"], "id": call_id, "args": {**base, "ticker": ticker}})
        groups[tc["id"]] = {"ids": ids, "skipped": skipped}
    return internal, groups


def chat_tool_node(state: ChatState) -> dict:
    """
    Executes every tool call from the last AIMessage.

    Parallel by default via ThreadPoolExecutor — these are all network-bound,
    so concurrency is close to free and cuts perceived latency a lot when the
    agent fans out (e.g. two companies' income statements at once).

    A financial_data_api call that carries a `tickers` list is expanded here into
    one parallel fetch per company (see _expand_batch_calls), so a multi-company
    question needs ONE model round however many companies it names.

    Exception: CHAT_SERIAL_TOOLS (Alpha Vantage) run one at a time, because
    they share a per-minute rate limiter and firing them concurrently just
    converts a slow success into a fast 429.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last = messages[-1]
    tool_calls = list(getattr(last, "tool_calls", None) or [])
    if not tool_calls:
        return {}

    internal_calls, groups = _expand_batch_calls(tool_calls)
    parallel = [tc for tc in internal_calls if tc["name"] not in CHAT_SERIAL_TOOLS]
    serial = [tc for tc in internal_calls if tc["name"] in CHAT_SERIAL_TOOLS]

    results: list[dict] = []

    # Failure memory for this turn, rebuilt from state (both lists are reset
    # every turn by the intent router).
    disabled = frozenset(state.get("disabled_tools") or [])
    failed_calls = frozenset(
        _call_key(t["tool"], t.get("args") or {})
        for t in (state.get("tool_trace") or [])
        if t.get("status") == "failed" and t.get("error_class") == BAD_ARGS
    )
    run_one = partial(_run_one, disabled=disabled, failed_calls=failed_calls)

    if parallel:
        workers = min(len(parallel), CHAT_MAX_PARALLEL_TOOL_CALLS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results.extend(pool.map(run_one, parallel))

    for tool_call in serial:
        results.append(run_one(tool_call))

    # Restore the model's original ordering so the ToolMessages line up with
    # the tool_calls array in the order the model expects to read them.
    by_id = {r["tool_call_id"]: r for r in results}
    ordered = [by_id[tc["id"]] for tc in internal_calls if tc["id"] in by_id]
    grouped_ids = {cid for g in groups.values() for cid in g["ids"]}

    message_by_id: dict[str, BaseMessage] = {}
    trace: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []

    for result in ordered:
        content = result["content"]
        truncated = len(content) > CHAT_TOOL_RESULT_MAX_CHARS
        if truncated:
            content = (
                content[:CHAT_TOOL_RESULT_MAX_CHARS]
                + "\n\n[...truncated for chat context. Ask for a full report if "
                "you need the complete filing text.]"
            )

        # Members of a batch are NOT given a ToolMessage of their own; they are
        # merged into the one message for the call the model actually made.
        if result["tool_call_id"] not in grouped_ids:
            message_by_id[result["tool_call_id"]] = ToolMessage(
                content=content, tool_call_id=result["tool_call_id"], name=result["tool"]
            )

        print(
            f"[ChatTools] {result['tool']} -> {result['status']} "
            f"({result['duration_s']}s, {len(result['content'])} chars)"
        )

        trace.append(
            {
                "tool": result["tool"],
                "args": result["args"],
                "status": result["status"],
                "error_class": result.get("error_class"),
                "retries": result.get("retries", 0),
                "duration_s": result["duration_s"],
                "chars": len(result["content"]),
                "truncated": truncated,
            }
        )

        # Preserve the FULL, untrimmed payload of successful calls so a later
        # synthesis failure can still show the user real data. Note this
        # stores result["content"], not the truncated `content` above — the
        # truncation exists to protect the LLM's context window, and has no
        # bearing on what a human can usefully be shown. For a batch this is
        # per COMPANY, so contradiction detection, the rescue prompt and the
        # evaluation judge all still see each company separately.
        if result["status"] == "success":
            preserved.append(
                {
                    "tool": result["tool"],
                    "args": result["args"],
                    "result": result["content"],
                    "status": "success",
                }
            )

    # One merged message per batch call. The companies share one context budget
    # (twice the single-result cap, split fairly) because eight full statements
    # at the single-result cap would overflow the token window and be dropped.
    from tools import merge_batch_results
    for call_id, group in groups.items():
        members = [by_id[cid] for cid in group["ids"] if cid in by_id]
        bodies = fair_share([m["content"] for m in members], CHAT_TOOL_RESULT_MAX_CHARS * 2)
        entries = [
            {"ticker": m["args"].get("ticker", "?"), "status": m["status"], "content": body}
            for m, body in zip(members, bodies)
        ]
        message_by_id[call_id] = ToolMessage(
            content=merge_batch_results(entries, group["skipped"]),
            tool_call_id=call_id, name="financial_data_api",
        )
        print(f"[ChatTools] financial_data_api batch of {len(members)} companies merged into one reply")

    tool_messages = [message_by_id[tc["id"]] for tc in tool_calls if tc["id"] in message_by_id]

    # Withdraw tools that just failed systemically. Never None: with the
    # reset_or_append reducer None would CLEAR the list, [] appends nothing.
    newly_disabled = sorted({
        r["tool"] for r in ordered
        if r["status"] == "failed" and r.get("error_class") == SYSTEMIC
        and r["tool"] not in disabled and r["tool"] in TOOLS_BY_NAME
    })
    if newly_disabled:
        print(f"[ChatTools] Withdrawing for the rest of this turn: {', '.join(newly_disabled)}")
    return {
        "messages": tool_messages,
        "tool_trace": trace,
        "successful_tool_results": preserved,
        "disabled_tools": newly_disabled,
        "tool_loops": state.get("tool_loops", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Cheap side nodes
# ─────────────────────────────────────────────────────────────────────────────

_SMALLTALK_PROMPT = """You are a financial research assistant. The user sent a
greeting, a thanks, or a question about your capabilities — no financial data
is needed. Reply in at most 4 short sentences, warm but not gushing.

If they asked what you can do, say concretely: pull financial statements and
ratios (yfinance / Alpha Vantage), read SEC filings and XBRL data, fetch
earnings call transcripts, search recent news, pull FRED macro series, and run
exact metric calculations. Mention that you can also generate a full
downloadable research report with charts if they ask for one.

Do not invent any numbers. Do not list your tools as a long bulleted menu."""


_OUT_OF_SCOPE_PROMPT = """You are a financial research assistant. The user asked something
outside finance. In at most 3 short sentences: if it is harmless and trivial you may
answer it very briefly, then say plainly that you focus on financial research and
invite a question about companies, filings, earnings, valuation or macro data.
Do not invent facts or numbers."""

_OUT_OF_SCOPE_FALLBACK = (
    "That's outside what I do — I'm a financial research assistant. Ask me about a "
    "company's financials, filings, earnings, valuation, or macro data and I'll "
    "pull real numbers."
)


def smalltalk_node(state: ChatState) -> dict:
    """
    Handles greetings without binding tools or loading the full analyst prompt.

    Worth its own node: it turns "hi" from a ~2500-token request with 6 tool
    schemas attached into a ~200-token one. On a free tier with a tokens-per-
    minute cap, that difference is the difference between a demo that keeps
    responding and one that 429s halfway through.
    """
    history = _window_messages(state.get("messages", []))[-6:]
    llm = get_llm(temperature=0.4, max_tokens=300)

    # The understanding step judged the message not to be about finance: same cheap
    # path as small talk (no tools, no analyst prompt), but it redirects the user.
    out_of_scope = state.get("intent") == INTENT_OUT_OF_SCOPE
    prompt = _OUT_OF_SCOPE_PROMPT if out_of_scope else _SMALLTALK_PROMPT

    try:
        response = invoke_with_retry(llm, [SystemMessage(content=prompt)] + history)
        return {"messages": [response]}
    except Exception as exc:
        print(f"[Smalltalk] LLM failed: {exc}")
        if out_of_scope:
            return {"messages": [AIMessage(content=_OUT_OF_SCOPE_FALLBACK)]}
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Hi — I'm a financial research assistant. I can pull "
                        "financial statements, SEC filings, earnings transcripts, "
                        "news and macro data, run exact metric calculations, and "
                        "generate a full research report if you want one. What "
                        "company are you looking at?"
                    )
                )
            ]
        }


def report_handoff_node(state: ChatState) -> dict:
    """
    Terminal node for a confirmed report request.

    It does NOT run the report pipeline. The pipeline is a multi-minute job and
    running it inside a request-scoped graph invocation would hold the HTTP
    connection open the whole time and lose all progress if the client blinks.
    Instead this emits the acknowledgement, and the API layer sees
    report_confirmed=True in the resulting state and dispatches the pipeline as
    a background job the client polls. Clean separation: the graph owns the
    conversation, the API owns long-running work.
    """
    query = state.get("pending_report_query") or state.get("user_input", "")
    report_type = state.get("pending_report_type") or "COMPANY_RESEARCH"

    print(f"[ReportHandoff] Confirmed: type={report_type} query={query!r}")

    return {
        "messages": [
            AIMessage(
                content=(
                    f"Starting the **{report_type}** report now. I'll fetch "
                    f"filings, financials and news, compute the metrics, and "
                    f"assemble the document with charts — this usually takes a "
                    f"few minutes.\n\n"
                    f"You'll get a download link when it's done. Feel free to keep "
                    f"asking me questions in the meantime."
                )
            )
        ],
        "report_confirmed": True,
    }
