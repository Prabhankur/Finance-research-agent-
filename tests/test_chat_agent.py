"""
tests/test_chat_agent.py  –  Offline tests for the chat agent graph.

No API keys, no network. The Groq client and the tool registry are both
stubbed, so these run in under a second and can't be broken by a rate limit
or a flaky data provider.

What's actually being tested is the CONTROL FLOW, which is the part that
regresses silently: does the router send each intent to the right node, does
the tool loop terminate at its budget, does history windowing avoid orphaning
tool messages, does the report handoff stay behind a confirmation.

Model output quality is deliberately not tested here — that needs the
evaluation benchmark from the handoff doc, which is a separate piece of work
with ground-truth data.

Run:
    python -m tests.test_chat_agent
    pytest tests/test_chat_agent.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import config
import utils.groq_client as groq_client

# ─────────────────────────────────────────────────────────────────────────────
#  Stubs
# ─────────────────────────────────────────────────────────────────────────────


class FakeLLM:
    """
    Returns pre-scripted responses in order. bind_tools is a no-op that records
    whether tools were bound, which is how we assert that the final forced turn
    really did drop them.
    """

    def __init__(self, script: list[AIMessage]):
        self.script = list(script)
        self.calls: list[list] = []
        self.tools_bound: list[bool] = []
        self.tool_choices: list = []
        self._has_tools = False
        self._tool_choice = None

    def bind_tools(self, tools, **kwargs):
        self._has_tools = True
        self._tool_choice = kwargs.get("tool_choice")
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        self.tools_bound.append(self._has_tools)
        self.tool_choices.append(self._tool_choice)
        if not self.script:
            return AIMessage(content="fallback answer")
        return self.script.pop(0)


def install_fake_llm(script: list[AIMessage]) -> FakeLLM:
    """Patches get_llm everywhere it's already been imported by name."""
    fake = FakeLLM(script)

    def _get_llm(*args, **kwargs):
        fake._has_tools = bool(kwargs.get("with_tools"))
        fake._tool_choice = kwargs.get("tool_choice")
        return fake

    groq_client.get_llm = _get_llm
    groq_client.invoke_with_retry = lambda llm, messages, **kw: llm.invoke(messages)

    import nodes.chat_agent as chat_agent_module
    import nodes.intent_router as intent_router_module
    import nodes.summarizer as summarizer_module

    chat_agent_module.get_llm = _get_llm
    chat_agent_module.invoke_with_retry = groq_client.invoke_with_retry
    intent_router_module.get_llm = _get_llm
    intent_router_module.invoke_with_retry = groq_client.invoke_with_retry
    summarizer_module.get_llm = _get_llm
    summarizer_module.invoke_with_retry = groq_client.invoke_with_retry

    return fake


class FakeTool:
    def __init__(self, name: str, output: str):
        self.name = name
        self.output = output
        self.call_count = 0

    def invoke(self, args):
        self.call_count += 1
        return self.output


def install_fake_tools(outputs: dict[str, str]) -> dict[str, FakeTool]:
    import nodes.chat_agent as chat_agent_module

    fakes = {name: FakeTool(name, out) for name, out in outputs.items()}
    chat_agent_module.TOOLS_BY_NAME = fakes
    return fakes


def fresh_graph():
    """
    A graph with its own IN-MEMORY checkpointer, isolated from every other
    test and from the real app.

    Necessary as of the SqliteSaver switch: build_chat_graph()'s default
    checkpointer now points at the real output/chat_checkpoints.db file (see
    graph/chat_graph.py) so conversations survive a server restart. Tests
    must NOT share that file — thread ids like "t1"/"m1" repeat across test
    functions and across repeated test runs, and a file-backed checkpointer
    would silently resume stale state from a previous run instead of giving
    each test a clean slate, which is exactly the kind of flakiness that's
    hard to diagnose later. Explicitly passing MemorySaver() keeps tests
    hermetic regardless of what the production default is.
    """
    from graph.chat_graph import build_chat_graph
    from langgraph.checkpoint.memory import MemorySaver

    return build_chat_graph(checkpointer=MemorySaver())


def run(graph, text: str, thread: str = "t1") -> dict:
    config_dict = {"configurable": {"thread_id": thread}, "recursion_limit": 50}
    graph.invoke({"messages": [HumanMessage(content=text)]}, config_dict)
    return graph.get_state(config_dict).values


def last_answer(values: dict) -> str:
    for message in reversed(values["messages"]):
        if isinstance(message, AIMessage) and not (getattr(message, "tool_calls", None) or []):
            return str(message.content)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")


def test_smalltalk_skips_tools():
    install_fake_llm([AIMessage(content="Hi! I can pull financial data for you.")])
    tools = install_fake_tools({"financial_data_api": "unused"})
    values = run(fresh_graph(), "hey")

    check("smalltalk routes to SMALLTALK", values["intent"] == config.INTENT_SMALLTALK,
          f"got {values.get('intent')}")
    check("smalltalk calls no tools", all(t.call_count == 0 for t in tools.values()))
    check("smalltalk answers", "financial data" in last_answer(values).lower())


def test_tool_loop_executes_and_returns():
    tool_call = {"name": "financial_data_api", "args": {"ticker": "AAPL"}, "id": "call_1"}
    install_fake_llm([
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="Apple's revenue was $391.0B (yfinance)."),
    ])
    tools = install_fake_tools({"financial_data_api": "Total Revenue: 391035000000"})

    values = run(fresh_graph(), "what is Apple's revenue?")

    check("chat intent for a question", values["intent"] == config.INTENT_CHAT,
          f"got {values.get('intent')}")
    check("tool was executed", tools["financial_data_api"].call_count == 1)
    check("tool_trace recorded", len(values.get("tool_trace", [])) == 1)
    check("trace marks success", values["tool_trace"][0]["status"] == "success",
          str(values.get("tool_trace")))
    check("answer uses tool data", "391" in last_answer(values))
    check("ticker stuck to state", values.get("active_ticker") == "AAPL",
          f"got {values.get('active_ticker')}")


def test_loop_budget_forces_an_answer():
    """A model that never stops asking for tools must still terminate."""
    loops = config.CHAT_MAX_TOOL_LOOPS
    endless = [
        AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": f"c{i}"}],
        )
        for i in range(loops + 5)
    ]
    fake = install_fake_llm(endless)
    tools = install_fake_tools({"web_search": "some result"})

    values = run(fresh_graph(), "tell me everything about the market today please")

    check("tool calls capped at budget", tools["web_search"].call_count <= loops,
          f"ran {tools['web_search'].call_count}, budget {loops}")
    check("turn terminated", values.get("tool_loops", 0) <= loops)
    # Phase 2 change: the final round now binds tools WITH an explicit
    # tool_choice="none" rather than omitting them. Omitting tools relied on
    # Groq's server-side default to disable tool calls, and gpt-oss models
    # were observed emitting tool calls anyway — including their own native
    # `web_open` — which Groq then 400'd. So the assertion is no longer
    # "no tools bound" but "tool calls were explicitly forbidden".
    check("last agent call forbade tool use via tool_choice",
          fake.tool_choices and fake.tool_choices[-1] == "none",
          f"tool_choices={fake.tool_choices}")


def test_report_requires_confirmation():
    install_fake_llm([AIMessage(content="unused")])
    tools = install_fake_tools({"financial_data_api": "unused"})
    graph = fresh_graph()

    values = run(graph, "generate a full research report on Apple", thread="r1")
    check("report intent detected", values["intent"] == config.INTENT_REPORT,
          f"got {values.get('intent')}")
    check("report NOT auto-started", values.get("report_confirmed") is False)
    check("report staged as pending", bool(values.get("pending_report_query")))
    check("no tools burned on the offer", all(t.call_count == 0 for t in tools.values()))
    check("offer asks for confirmation", "yes" in last_answer(values).lower())

    values = run(graph, "yes", thread="r1")
    check("yes confirms report", values.get("report_confirmed") is True)
    check("confirm intent", values["intent"] == config.INTENT_CONFIRM_REPORT,
          f"got {values.get('intent')}")


def test_bare_yes_without_pending_offer_is_not_a_report():
    """A stray 'yes' must not resurrect a report request."""
    install_fake_llm([AIMessage(content="Sure — what would you like to know?")])
    install_fake_tools({})
    values = run(fresh_graph(), "yes", thread="y1")
    check("bare yes is not CONFIRM_REPORT",
          values["intent"] != config.INTENT_CONFIRM_REPORT,
          f"got {values.get('intent')}")
    check("bare yes does not confirm", values.get("report_confirmed") is False)


def test_memory_persists_across_turns():
    install_fake_llm([
        AIMessage(content="Apple trades at a P/E of 32 (yfinance)."),
        AIMessage(content="Its gross margin is 46% (yfinance)."),
    ])
    install_fake_tools({})
    graph = fresh_graph()

    run(graph, "what is Apple's P/E?", thread="m1")
    values = run(graph, "and its margins?", thread="m1")

    human_turns = [m for m in values["messages"] if isinstance(m, HumanMessage)]
    check("both turns retained", len(human_turns) == 2, f"got {len(human_turns)}")
    check("ticker carried into follow-up", values.get("active_ticker") == "AAPL",
          f"got {values.get('active_ticker')}")


def test_ticker_extraction_ignores_stopwords():
    from nodes.intent_router import extract_ticker_from_text

    cases = [
        ("IS AAPL A BUY", "AAPL"),
        ("how is apple doing", "AAPL"),
        ("what about TSLA", "TSLA"),
        ("compare tcs and infosys", "TCS.NS"),
        ("what is the GDP outlook", None),
        ("tell me about the market", None),
    ]
    for text, expected in cases:
        got = extract_ticker_from_text(text)
        check(f"ticker({text!r}) -> {expected}", got == expected, f"got {got}")


def test_history_window_never_orphans_tool_messages():
    from nodes.chat_agent import _window_messages

    # An AI tool-call message whose ToolMessage answers get cut by the window.
    messages = (
        [HumanMessage(content=f"filler {i}") for i in range(config.CHAT_HISTORY_MAX_MESSAGES)]
        + [AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "z1"}])]
        + [ToolMessage(content="result", tool_call_id="z1", name="web_search")]
    )
    windowed = _window_messages(messages)

    check("window respects the cap", len(windowed) <= config.CHAT_HISTORY_MAX_MESSAGES)
    check("no leading orphan ToolMessage", not isinstance(windowed[0], ToolMessage))

    # A window that starts mid-pair must drop the whole pair, not half of it.
    tool_ids = {m.tool_call_id for m in windowed if isinstance(m, ToolMessage)}
    requested = {
        tc["id"]
        for m in windowed
        if isinstance(m, AIMessage)
        for tc in (getattr(m, "tool_calls", None) or [])
    }
    check("every ToolMessage has its parent", tool_ids.issubset(requested),
          f"orphans={tool_ids - requested}")


def test_tool_failure_is_surfaced_not_swallowed():
    tool_call = {"name": "sec_filing_search", "args": {"ticker": "ZZZZ"}, "id": "f1"}
    install_fake_llm([
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="I couldn't retrieve that filing."),
    ])
    install_fake_tools({"sec_filing_search": "[SEC] Error: unauthorized"})

    values = run(fresh_graph(), "get me ZZZZ's 10-K", thread="e1")
    trace = values.get("tool_trace", [])
    check("failure recorded in trace", trace and trace[0]["status"] == "failed",
          str(trace))
    check("failure classified", trace and trace[0]["error_class"] == "systemic",
          str(trace))


def test_report_type_override_by_typing():
    """User overrides the classifier's guess by naming a different type."""
    install_fake_llm([AIMessage(content="unused")])
    install_fake_tools({})
    graph = fresh_graph()

    values = run(graph, "generate a full research report on Apple", thread="ov1")
    check("offer stages a default type", bool(values.get("pending_report_type")))

    values = run(graph, "generate it as a Valuation / Risk Deep-Dive report", thread="ov1")
    check("override applied", values.get("pending_report_type") == "VALUATION_RISK",
          f"got {values.get('pending_report_type')}")
    check("override still confirms", values.get("report_confirmed") is True)


def test_report_type_classifier_gets_ticker_on_fallback():
    """
    Regression test for the bug where the classifier fallback always saw an
    empty ticker (state dict was missing the key entirely), so any LLM
    failure silently produced OTHER even for a query that plainly named a
    company. Forces the classifier's LLM call to fail and checks the
    fallback lands on COMPANY_RESEARCH, not OTHER, when a ticker is present.
    """
    from nodes.intent_router import _classify_report_type

    def _always_fails(*a, **k):
        raise RuntimeError("simulated Groq rate limit")

    import nodes.report_type_classifier as classifier_module
    original = classifier_module.get_llm
    classifier_module.get_llm = _always_fails
    try:
        result = _classify_report_type("analyze microsoft for 2026", ticker="MSFT")
        check("fallback uses passed-in ticker, not OTHER", result == "COMPANY_RESEARCH",
              f"got {result}")
    finally:
        classifier_module.get_llm = original


def test_token_windowing_trims_large_tool_results():
    """
    Regression test for the BadRequestError: a comparison-style turn where
    several large tool results fit comfortably under the MESSAGE-count cap
    but should NOT fit under the TOKEN cap. Confirms the token pass actually
    trims, and confirms it never leaves an orphaned ToolMessage behind.
    """
    from nodes.chat_agent import _window_messages, _count_tokens
    import config

    # A handful of ~5000-char tool results (chat's own per-result cap) — the
    # exact shape that triggered this in production on multi-entity queries.
    big_result = "x" * config.CHAT_TOOL_RESULT_MAX_CHARS
    messages: list = [HumanMessage(content="compare NVDA, MSFT, AMD")]
    for i in range(6):
        call_id = f"c{i}"
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "financial_data_api", "args": {}, "id": call_id}],
            )
        )
        messages.append(ToolMessage(content=big_result, tool_call_id=call_id, name="financial_data_api"))
    messages.append(HumanMessage(content="now explain the differences"))

    total_before = sum(_count_tokens(str(m.content)) for m in messages)
    check("test setup exceeds message-count-only assumptions",
          len(messages) < config.CHAT_HISTORY_MAX_MESSAGES,
          f"len={len(messages)}")
    check("test setup genuinely exceeds token budget",
          total_before > config.CHAT_MAX_CONTEXT_TOKENS,
          f"{total_before} tokens vs budget {config.CHAT_MAX_CONTEXT_TOKENS}")

    windowed = _window_messages(messages)
    total_after = sum(_count_tokens(str(m.content)) for m in windowed)

    check("token pass actually trims under budget (with pairing slack)",
          total_after <= config.CHAT_MAX_CONTEXT_TOKENS + 2000,
          f"got {total_after}")
    check("token pass removed messages", len(windowed) < len(messages),
          f"before={len(messages)} after={len(windowed)}")
    check("no orphan ToolMessage after token trim",
          not (windowed and isinstance(windowed[0], ToolMessage)))

    tool_ids = {m.tool_call_id for m in windowed if isinstance(m, ToolMessage)}
    requested = {
        tc["id"] for m in windowed if isinstance(m, AIMessage)
        for tc in (getattr(m, "tool_calls", None) or [])
    }
    check("every remaining ToolMessage still has its parent",
          tool_ids.issubset(requested), f"orphans={tool_ids - requested}")


def test_gross_margin_accepts_gross_profit():
    """
    Regression test for the exact production crash: yfinance/AV return
    gross_profit, not cogs, and the old calculator signature only recognized
    cogs — silently dropping gross_profit and crashing with 'missing 1
    required positional argument'.
    """
    from tools.calculator import calculator

    # The real numbers from the user's MSFT trace.
    result = calculator(metric="gross_margin", revenue=331839000000, gross_profit=225465000000)
    check("gross_margin computes from gross_profit", "68" in result or "67.9" in result,
          result)
    check("no crash / no missing-argument error", "Missing" not in result and "Error" not in result,
          result)

    # cogs path still works (backward compatible).
    result_cogs = calculator(metric="gross_margin", revenue=100, cogs=60)
    check("gross_margin still computes from cogs", "40.00%" in result_cogs, result_cogs)

    # Neither given: clear error, not a crash.
    result_neither = calculator(metric="gross_margin", revenue=100)
    check("missing both args gives a clean error, not a stack trace",
          "Error" in result_neither, result_neither)


def test_summarizer_skips_short_history():
    """Below CHAT_SUMMARIZE_MIN_MESSAGES, no LLM call, no state change."""
    from nodes.summarizer import summarizer_node

    fake = install_fake_llm([AIMessage(content="should not be called")])
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}
    result = summarizer_node(state)

    check("short history: no-op update", result == {}, str(result))
    check("short history: LLM never invoked", len(fake.calls) == 0, str(len(fake.calls)))


def test_summarizer_folds_old_messages_and_sets_summary():
    """
    Builds a history long enough that _window_messages() would keep only the
    tail — confirms the head gets folded into a summary AND actually removed
    from state via RemoveMessage (not just excluded from one prompt).
    """
    from langchain_core.messages import RemoveMessage
    from nodes.summarizer import summarizer_node
    import config

    fake = install_fake_llm([
        AIMessage(content="User asked about AAPL revenue ($391B FY2024) and MSFT margins.")
    ])

    # Build enough old turns to exceed the keep-window, each with a real id
    # (RemoveMessage needs one — see nodes/summarizer.py).
    old_messages = []
    for i in range(config.CHAT_HISTORY_MAX_MESSAGES + 10):
        msg = HumanMessage(content=f"old turn {i}") if i % 2 == 0 else AIMessage(content=f"old reply {i}")
        msg.id = f"old-{i}"
        old_messages.append(msg)
    recent = HumanMessage(content="what about now")
    recent.id = "recent-1"

    state = {"messages": old_messages + [recent]}
    result = summarizer_node(state)

    check("summary was set", bool(result.get("conversation_summary")), str(result))
    check("LLM was called exactly once", len(fake.calls) == 1, str(len(fake.calls)))

    removed = result.get("messages", [])
    check("removal instructions returned", len(removed) > 0, str(len(removed)))
    check("removals are RemoveMessage objects", all(isinstance(m, RemoveMessage) for m in removed))

    removed_ids = {m.id for m in removed}
    check("the most recent message is NEVER removed", "recent-1" not in removed_ids,
          str(removed_ids))
    check("some old messages ARE removed", any(mid.startswith("old-") for mid in removed_ids),
          str(removed_ids))


def test_summarizer_merges_with_existing_summary():
    """A prior summary must be passed to the LLM, not silently discarded."""
    from nodes.summarizer import summarizer_node
    import config

    fake = install_fake_llm([AIMessage(content="merged summary text")])

    old_messages = []
    for i in range(config.CHAT_HISTORY_MAX_MESSAGES + 6):
        msg = HumanMessage(content=f"turn {i}")
        msg.id = f"m-{i}"
        old_messages.append(msg)

    state = {
        "messages": old_messages,
        "conversation_summary": "Earlier: user discussed TSLA risk factors.",
    }
    summarizer_node(state)

    sent_content = str(fake.calls[0][1].content)  # [SystemMessage, HumanMessage]
    check("existing summary reached the LLM call", "TSLA risk factors" in sent_content,
          sent_content[:200])
    check("marked as an existing summary, not new messages",
          "EXISTING SUMMARY" in sent_content, sent_content[:200])


def test_summarizer_failure_does_not_break_the_turn():
    """If the summarization LLM call itself fails, the turn must survive —
    summarization is a nice-to-have, not load-bearing."""
    from nodes.summarizer import summarizer_node
    import config

    def _boom(*a, **k):
        raise RuntimeError("simulated Groq outage")

    import nodes.summarizer as summarizer_module
    original = summarizer_module.get_llm
    summarizer_module.get_llm = _boom
    try:
        old_messages = []
        for i in range(config.CHAT_HISTORY_MAX_MESSAGES + 6):
            msg = HumanMessage(content=f"turn {i}")
            msg.id = f"f-{i}"
            old_messages.append(msg)

        result = summarizer_node({"messages": old_messages})
        check("failure returns a safe no-op, not an exception", result == {}, str(result))
    finally:
        summarizer_module.get_llm = original


def test_chart_tool_trace_carries_full_untrimmed_args():
    """
    render_chart's payload (labels/series/values) must survive intact in the
    tool_trace — it's what api/server.py streams as the chart_data SSE event.
    Confirms args aren't accidentally caught by CHAT_TOOL_RESULT_MAX_CHARS
    trimming, which only applies to `content`, never to `args`.
    """
    from nodes.chat_agent import chat_tool_node

    series = [{"name": "Gross Margin %", "values": [75.0, 47.0, 68.0]}]
    tool_call = {
        "name": "render_chart",
        "args": {
            "chart_type": "bar",
            "title": "Gross Margin by Company",
            "labels": ["NVDA", "AMD", "MSFT"],
            "series": series,
        },
        "id": "chart1",
    }
    state = {"messages": [AIMessage(content="", tool_calls=[tool_call])], "tool_loops": 0}

    result = chat_tool_node(state)
    trace = result["tool_trace"][0]

    check("chart tool ran successfully", trace["status"] == "success", str(trace))
    check("full labels survived in trace args", trace["args"]["labels"] == ["NVDA", "AMD", "MSFT"],
          str(trace["args"]))
    check("full series survived in trace args", trace["args"]["series"] == series,
          str(trace["args"]))


def test_chart_tool_rejects_mismatched_series_length():
    """A real correctness check: series shorter/longer than labels must be
    caught by the tool itself, not silently mis-render on the frontend."""
    from tools.chart_tool import render_chart

    result = render_chart.invoke({
        "chart_type": "bar", "title": "Bad chart",
        "labels": ["NVDA", "AMD", "MSFT"],
        "series": [{"name": "Margin", "values": [75.0, 47.0]}],  # only 2, not 3
    })
    check("mismatched series length is rejected", "Error" in result, result)


def test_daily_rate_limit_fails_fast_without_sleeping():
    """
    Regression test for a real production incident: a tokens-PER-DAY (TPD)
    429 was being retried with the same seconds-scale backoff as a
    tokens-per-minute one, sleeping up to ~5 minutes total before failing
    anyway (a daily quota cannot reset within that window). Confirms TPD
    errors now raise immediately with zero sleep, and that a genuine TPM
    429 still retries normally (this must NOT become "never retry on 429").
    """
    import time
    from utils.groq_client import invoke_with_retry

    tpd_message = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for "
        "model openai/gpt-oss-120b ... on tokens per day (TPD): Limit "
        "200000, Used 199632, Requested 5054. Please try again in 23h58m.'}}"
    )

    class TPDLLM:
        def invoke(self, messages):
            raise Exception(tpd_message)

    started = time.time()
    try:
        invoke_with_retry(TPDLLM(), [], max_retries=5)
        check("TPD error should raise, not return", False, "no exception raised")
    except RuntimeError as exc:
        elapsed = time.time() - started
        check("TPD error raised immediately (no multi-minute sleep)", elapsed < 1.0,
              f"took {elapsed:.1f}s")
        check("TPD error message says retrying won't help",
              "cannot help" in str(exc) or "quota exhausted" in str(exc).lower(),
              str(exc))

    tpm_message = "Error code: 429 - rate_limit_exceeded. Please try again in 0.05s"
    call_count = {"n": 0}

    class TPMLLM:
        def invoke(self, messages):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise Exception(tpm_message)
            return "ok"

    result = invoke_with_retry(TPMLLM(), [], max_retries=5)
    check("genuine TPM 429 still retries and succeeds", result == "ok", str(result))
    check("TPM retry actually attempted twice", call_count["n"] == 2, str(call_count))


def test_registered_calculator_tool_accepts_gross_profit():
    """
    Regression test for a real deployed bug found by the user in production:
    tools/calculator.py's gross_margin() function was fixed to accept
    gross_profit, but tools/__init__.py defines a SEPARATE, independently
    declared @lc_tool wrapper — the one actually placed in ALL_TOOLS /
    TOOLS_BY_NAME and exposed to the LLM (both in chat and the report
    pipeline) — with its OWN parameter list that was never updated. LangChain
    builds each tool's schema from its OWN function signature, so the
    registered wrapper's schema had no gross_profit field at all, and
    silently dropped it before tools/calculator.py's real function ever saw
    it — reproducing the exact symptom reported: gross_profit given in the
    call, but the function still says neither cogs nor gross_profit was
    provided.

    The earlier test (test_gross_margin_accepts_gross_profit) only exercised
    tools.calculator.calculator directly, which is NOT the tool the agent
    actually calls — passing it proved nothing about whether the real,
    registered tool worked. This test goes through `from tools import
    calculator`, the exact object in TOOLS_BY_NAME, exactly as
    chat_tool_node and the report pipeline's compute executor call it.
    """
    from tools import calculator as registered_calculator

    # Catches a schema-only regression even before invoking anything — this
    # check alone would have caught the actual bug immediately.
    schema_fields = getattr(registered_calculator.args_schema, "model_fields", {})
    check("the REGISTERED tool's schema declares gross_profit",
          "gross_profit" in schema_fields, f"schema has: {list(schema_fields.keys())}")

    # The user's exact real-world numbers.
    result = registered_calculator.invoke({
        "metric": "gross_margin",
        "revenue": 130497000000,
        "gross_profit": 97858000000,
    })
    check("registered tool computes gross_margin from gross_profit (no error)",
          "Error" not in result, result)
    check("registered tool's computed value is correct (~74.99%)",
          "74.9" in result, result)


def test_sec_xbrl_revenue_uses_modern_tag_not_stale_one():
    """
    Regression test for the exact production bug: Apple's FY2024 revenue
    request returned $265.6B (Apple's real FY2018 figure) because the old
    'Revenues' XBRL tag has no facts past ~2018 (ASC 606 tag change), and the
    code used to silently substitute the newest fact it COULD find under
    that one tag, mislabeled as if it matched the requested year.
    """
    from unittest.mock import patch
    from tools.sec_filing import _fetch_xbrl_facts

    fake_facts = {
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                {"end": "2018-09-29", "val": 265595000000, "form": "10-K", "fp": "FY"},
            ]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                {"end": "2024-09-28", "val": 391035000000, "form": "10-K", "fp": "FY"},
            ]}},
        }}
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return fake_facts

    with patch("tools.sec_filing.requests.get", return_value=FakeResp()):
        result = _fetch_xbrl_facts(cik="0000320193", ticker="AAPL", year=2024, filing_type="10-K")

    check("FY2024 revenue uses the modern tag's $391B, not stale FY2018 $265.6B",
          "391" in result, result)
    check("stale FY2018 figure does NOT appear as the shown revenue",
          "265.5" not in result.split("Total Revenue")[1].split("\n")[0] if "Total Revenue" in result else True,
          result)


def test_sec_xbrl_never_silently_substitutes_wrong_year():
    """A requested year with genuinely no data anywhere must say so plainly,
    not silently show whatever the most recent available fact happens to
    be as if it matched."""
    from unittest.mock import patch
    from tools.sec_filing import _fetch_xbrl_facts

    fake_facts = {
        "facts": {"us-gaap": {
            "NetIncomeLoss": {"units": {"USD": [
                {"end": "2024-09-28", "val": 93736000000, "form": "10-K", "fp": "FY"},
            ]}},
        }}
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return fake_facts

    with patch("tools.sec_filing.requests.get", return_value=FakeResp()):
        result = _fetch_xbrl_facts(cik="0000320193", ticker="AAPL", year=1999, filing_type="10-K")

    check("unavailable year is explicitly flagged as N/A for that year",
          "N/A for FY1999" in result, result)
    check("the 2024 figure is NOT silently shown as if it were FY1999",
          "93.74B" not in result or "N/A for FY1999" in result, result)


def test_synthesis_failure_preserves_fetched_data():
    """
    THE core Phase 1 guarantee: when tools succeed and the final synthesis
    LLM call then fails, the fetched data must be shown to the user, not
    discarded behind a bare "I hit an error". Real API calls cost real quota
    against a finite daily limit — throwing their results away because the
    write-up step 400'd is the single worst failure mode this agent had.

    Simulates exactly the production scenario: financial_data_api succeeds,
    then the synthesis call raises the Groq tool_use_failed 400.
    """
    real_data = (
        "[FinData] Income Statement - MSFT\n"
        "Total Revenue: 331839000000\n"
        "Net Income: 133700000000\n"
        "Gross Profit: 225465000000\n"
    )

    class FailOnSynthesis:
        def __init__(self):
            self.n = 0
        def bind_tools(self, tools, **k):
            return self
        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                return AIMessage(content="", tool_calls=[
                    {"name": "financial_data_api", "args": {"ticker": "MSFT"}, "id": "d1"}
                ])
            raise Exception(
                "Error code: 400 - {'error': {'message': 'Tool choice is none, "
                "but model called a tool', 'code': 'tool_use_failed'}}"
            )

    fake = FailOnSynthesis()
    import utils.groq_client as gc
    import nodes.chat_agent as ca, nodes.intent_router as ir, nodes.summarizer as sm
    for module in (ca, ir, sm):
        module.get_llm = lambda *a, **k: fake
        module.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)

    install_fake_tools({"financial_data_api": real_data})
    values = run(fresh_graph(), "what is MSFT revenue?", thread="partial1")

    answer = last_answer(values)
    check("turn marked as synthesis-failed, not generic error",
          values.get("turn_status") == "RETRIEVAL_COMPLETE_SYNTHESIS_FAILED",
          f"got {values.get('turn_status')}")
    check("fetched revenue figure survives the failure", "331839000000" in answer or "331.84" in answer,
          answer[:400])
    check("answer is explicitly labelled incomplete",
          "incomplete" in answer.lower() or "partial" in answer.lower(), answer[:300])
    check("answer does NOT pretend to be finished analysis",
          "not a finished analysis" in answer.lower() or "raw retrieved" in answer.lower(),
          answer[:400])
    check("technical cause is still surfaced", "tool_use_failed" in answer, answer[-400:])


def test_synthesis_failure_with_no_data_gives_plain_error():
    """Nothing fetched means nothing to salvage — must not fabricate a
    'partial results' message with an empty body."""
    class AlwaysFails:
        def bind_tools(self, tools, **k):
            return self
        def invoke(self, messages):
            raise Exception("Error code: 500 - internal server error")

    fake = AlwaysFails()
    import nodes.chat_agent as ca, nodes.intent_router as ir, nodes.summarizer as sm
    for module in (ca, ir, sm):
        module.get_llm = lambda *a, **k: fake
        module.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)

    install_fake_tools({})
    values = run(fresh_graph(), "what is AAPL revenue?", thread="nodata1")

    answer = last_answer(values)
    check("turn marked RETRIEVAL_FAILED", values.get("turn_status") == "RETRIEVAL_FAILED",
          f"got {values.get('turn_status')}")
    check("plain error, not a fake partial-results block",
          "Partial result" not in answer, answer[:300])


def test_preserved_results_reset_between_turns():
    """
    A synthesis failure on turn 2 must not replay turn 1's tool results as
    though they answered the new question. Guards the reset_or_append
    reducer wiring (see graph/chat_state.py).
    """
    class ToolThenFail:
        def __init__(self):
            self.turn = 0
            self.n = 0
        def bind_tools(self, tools, **k):
            return self
        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                return AIMessage(content="", tool_calls=[
                    {"name": "financial_data_api", "args": {"ticker": "AAPL"}, "id": "t1"}
                ])
            if self.n == 2:
                return AIMessage(content="Apple revenue is $391B.")
            raise Exception("Error code: 400 - synthesis blew up")

    fake = ToolThenFail()
    import nodes.chat_agent as ca, nodes.intent_router as ir, nodes.summarizer as sm
    for module in (ca, ir, sm):
        module.get_llm = lambda *a, **k: fake
        module.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)

    install_fake_tools({"financial_data_api": "AAPL_SENTINEL_REVENUE 391035000000"})
    graph = fresh_graph()

    run(graph, "AAPL revenue?", thread="reset1")           # turn 1 succeeds
    values = run(graph, "and NVDA?", thread="reset1")      # turn 2 fails, no tools ran

    answer = last_answer(values)
    check("turn 1's data is NOT replayed into turn 2's failure",
          "AAPL_SENTINEL_REVENUE" not in answer, answer[:400])


def test_formatter_scales_raw_figures():
    """The deterministic formatter must make raw provider integers readable
    without any LLM involvement."""
    from utils.result_formatter import format_number, build_partial_response

    check("trillions", format_number(4666000000000) == "$4.67T", format_number(4666000000000))
    check("billions", format_number(331839000000) == "$331.84B", format_number(331839000000))
    check("millions", format_number(45300000) == "$45.30M", format_number(45300000))
    check("non-numeric passes through", format_number("N/A") == "N/A")

    message = build_partial_response(
        [{"tool": "fred_series", "args": {"series_id": "DFF"}, "result": "Fed Funds Rate: 4.33"}],
        Exception("boom"),
    )
    check("formatter names the tool", "fred_series" in message, message[:200])
    check("formatter includes the value", "4.33" in message, message[:400])


def test_cache_prevents_duplicate_api_calls():
    """
    The core caching guarantee: an identical tool call within its TTL must
    NOT hit the live API a second time. This is what protects the Alpha
    Vantage 25-calls-per-day budget.
    """
    from database.cache_service import cached_call, make_cache_key

    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return "Total Revenue: 331839000000"

    args = {"ticker": f"CACHETEST{os.getpid()}", "years": 2}
    p1, hit1 = cached_call("financial_data_api", args, "annual_financials", fetch)
    p2, hit2 = cached_call("financial_data_api", args, "annual_financials", fetch)

    check("live API called exactly once", calls["n"] == 1, f"called {calls['n']}x")
    check("first call was a miss", hit1 is False)
    check("second call was a hit", hit2 is True)
    check("cached payload matches original", p1 == p2)

    # Argument ORDER must not change the key — an LLM will emit the same
    # call with keys in different orders and those must not count as two
    # separate cache entries.
    reordered = {"years": 2, "ticker": args["ticker"]}
    check("cache key is argument-order independent",
          make_cache_key("financial_data_api", args) == make_cache_key("financial_data_api", reordered))


def test_cache_never_stores_error_responses():
    """Caching a transient failure would turn a momentary blip into a
    TTL-long outage for that query."""
    from database.cache_service import cached_call

    calls = {"n": 0}

    def failing_fetch():
        calls["n"] += 1
        return "[FinData] Error: rate limit reached"

    args = {"ticker": f"ERRTEST{os.getpid()}"}
    cached_call("financial_data_api", args, "annual_financials", failing_fetch)
    _, hit = cached_call("financial_data_api", args, "annual_financials", failing_fetch)

    check("error response was NOT cached", hit is False)
    check("live API retried after an error", calls["n"] == 2, f"called {calls['n']}x")


def test_cache_failure_degrades_to_no_cache_not_crash():
    """A broken database must never take the agent down — it should just
    mean 'no caching'."""
    import database.cache_service as cs

    original_get = cs.get_cached
    original_set = cs.set_cached
    cs.get_cached = lambda key: (_ for _ in ()).throw(RuntimeError("db is on fire"))
    cs.set_cached = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db is on fire"))
    try:
        try:
            payload, hit = cs.cached_call(
                "financial_data_api", {"ticker": "X"}, "annual_financials",
                lambda: "live data",
            )
            check("broken cache still returns live data", payload == "live data", payload)
            check("broken cache reports a miss", hit is False)
        except Exception as exc:
            check("broken cache must not raise", False, f"raised {exc}")
    finally:
        cs.get_cached = original_get
        cs.set_cached = original_set


def test_normalization_writes_typed_rows_with_provenance():
    """
    THE Phase A guarantee: a yfinance response gets parsed into typed rows
    in income_statements + balance_sheets + cash_flow_statements, each row
    linked to a sources row that records provider and retrieved_at.
    """
    from unittest.mock import patch
    import tools
    from database.connection import query, query_one

    sample_text = (
        "[FinData] Income Statement (Annual) - PROVTEST\n"
        "Period: 2024-06-30\n"
        "  Total Revenue      : 245122000000\n"
        "  Gross Profit       : 178870000000\n"
        "  Net Income         : 88136000000\n"
        "Period: 2023-06-30\n"
        "  Total Revenue      : 211915000000\n"
        "  Net Income         : 72361000000\n"
    )

    with patch("tools._findata", return_value=sample_text):
        tools.financial_data_api_fn(ticker="PROVTEST", statement_type="income", years=2)

    company = query_one("SELECT id FROM companies WHERE ticker = ?", ("PROVTEST",))
    check("company row created", company is not None, str(company))

    rows = query(
        "SELECT * FROM income_statements WHERE company_id = ? ORDER BY period_end DESC",
        (company["id"],),
    )
    check("two income rows written", len(rows) == 2, f"got {len(rows)}")
    check("FY2024 revenue stored correctly", rows[0]["revenue"] == 245122000000,
          str(rows[0]))
    check("each row links to a source", all(r["source_id"] is not None for r in rows),
          str([r["source_id"] for r in rows]))

    source = query_one("SELECT * FROM sources WHERE id = ?", (rows[0]["source_id"],))
    check("source row records provider", source["provider"] == "yfinance", str(source))
    check("source row records retrieved_at", bool(source["retrieved_at"]))


def test_db_rebuild_serves_different_shaped_requests_without_api():
    """
    The API-reduction win over text-cache-only: two calls with different
    statement_type / years that would MISS the text cache are BOTH served
    from the DB after a single normalize.
    """
    from unittest.mock import patch
    import tools

    sample_text = (
        "[FinData] Income Statement (Annual) - DBTEST\n"
        "Period: 2024-06-30\n"
        "  Total Revenue      : 100\n"
        "  Net Income         : 20\n"
        "Period: 2023-06-30\n"
        "  Total Revenue      : 90\n"
        "  Net Income         : 15\n"
        "[FinData] Balance Sheet (Annual) - DBTEST\n"
        "Period: 2024-06-30\n"
        "  Total Assets       : 500\n"
        "  Total Liabilities  : 300\n"
        "  Shareholder Equity : 200\n"
        "  Cash & Equivalents : 50\n"
        "  Total Debt         : 100\n"
        "Period: 2023-06-30\n"
        "  Total Assets       : 400\n"
        "  Total Liabilities  : 250\n"
        "  Shareholder Equity : 150\n"
        "  Cash & Equivalents : 40\n"
        "  Total Debt         : 90\n"
        "[FinData] Cash Flow (Annual) - DBTEST\n"
        "Period: 2024-06-30\n"
        "  Operating Cash Flow: 30\n"
        "  CapEx              : -10\n"
        "  Free Cash Flow     : 20\n"
        "Period: 2023-06-30\n"
        "  Operating Cash Flow: 25\n"
        "  CapEx              : -8\n"
        "  Free Cash Flow     : 17\n"
    )

    api_calls = {"n": 0}
    def fake_fin(**kw):
        api_calls["n"] += 1
        return sample_text

    with patch("tools._findata", side_effect=fake_fin):
        tools.financial_data_api_fn(ticker="DBTEST", statement_type="all", years=2)
        # Different args — text cache MISS — but DB has the data
        r2 = tools.financial_data_api_fn(ticker="DBTEST", statement_type="income", years=2)
        r3 = tools.financial_data_api_fn(ticker="DBTEST", statement_type="balance", years=2)

    check("only one live API call across 3 differently-shaped requests",
          api_calls["n"] == 1, f"got {api_calls['n']}")
    check("differently-shaped request served from DB",
          "[FinData:DB]" in r2, r2[:120])
    check("DB-served response carries real data", "100" in r2, r2[:200])


def test_report_jobs_persist_across_restart():
    """
    UI bug fix regression: reports used to live in an in-memory dict wiped
    on every restart. Now backed by report_jobs table.
    """
    from database import report_jobs as rj

    rj.create("persist-1", "test query", "COMPANY_RESEARCH", thread_id="t-persist")
    rj.update("persist-1", status="completed", report_path="/tmp/x.md")

    got = rj.get("persist-1")
    check("job survives via DB", got is not None and got["status"] == "completed",
          str(got))
    thread_reports = rj.list_for_thread("t-persist")
    check("thread-scoped list works", len(thread_reports) == 1, str(thread_reports))


def test_field_filter_trims_response_but_keeps_structure():
    """The fields param should dramatically shrink the response — that's
    the whole point — while keeping section headers so it still parses."""
    from tools import _filter_fields

    full = (
        "[FinData] Market Overview – MSFT\n"
        "============================================================\n"
        "{\n"
        '  "Company Name": "Microsoft",\n'
        '  "Current Price": 465.44,\n'
        '  "Trailing P/E": 37.42,\n'
        '  "Revenue (TTM)": 245122000000,\n'
        '  "Net Income (TTM)": 88136000000,\n'
        '  "Beta": 0.94,\n'
        "}"
    )
    filtered = _filter_fields(full, ["revenue"])

    check("filtered response is smaller", len(filtered) < len(full),
          f"full={len(full)} filtered={len(filtered)}")
    check("filtered kept revenue line", "Revenue (TTM)" in filtered, filtered)
    check("filtered dropped beta", "Beta" not in filtered, filtered)
    check("filtered kept section header", "[FinData] Market Overview" in filtered, filtered)


def test_field_filter_falls_back_safely_on_unknown_field():
    """Unknown field aliases must NOT silently drop everything — safer to
    return the full response than to answer with nothing."""
    from tools import _filter_fields

    full = "[FinData] Test\n  Revenue: 100\n  Net Income: 50\n"
    result = _filter_fields(full, ["completely_unknown_thing"])
    check("unknown fields fall back to full response", result == full, result)


def test_field_filter_no_fields_returns_full_payload():
    """No fields = no filtering. Zero-cost default behavior."""
    from tools import _filter_fields

    full = "[FinData] Test\n  Revenue: 100\n"
    check("None fields returns full", _filter_fields(full, None) == full)
    check("empty list returns full", _filter_fields(full, []) == full)


def test_field_filter_handles_multiple_fields_and_periods():
    """Statement responses (multi-period) with multiple field aliases —
    the case that motivates this whole optimization."""
    from tools import _filter_fields

    full = (
        "[FinData] Income Statement (Annual) – MSFT\n"
        "Period: 2024-06-30\n"
        "  Total Revenue      : 245122000000\n"
        "  Gross Profit       : 178870000000\n"
        "  Operating Income   : 109433000000\n"
        "  Net Income         : 88136000000\n"
        "  EBITDA             : 130549000000\n"
        "Period: 2023-06-30\n"
        "  Total Revenue      : 211915000000\n"
        "  Gross Profit       : 146052000000\n"
        "  Operating Income   : 88523000000\n"
        "  Net Income         : 72361000000\n"
        "  EBITDA             : 105155000000\n"
    )
    filtered = _filter_fields(full, ["revenue", "operating_margin"])

    check("kept revenue lines", "Total Revenue" in filtered)
    check("kept operating income lines", "Operating Income" in filtered)
    check("dropped EBITDA", "EBITDA" not in filtered)
    check("kept both period headers",
          filtered.count("Period:") == 2, f"got {filtered.count('Period:')}")


def test_rescue_synthesis_recovers_when_retry_replays_bad_history():
    """
    Regression test for the exact production scenario: synthesis fails with
    'Tool choice is none, but model called a tool', AND groq_client's own
    internal retry (which replays the same tool-call-laden history) ALSO
    fails identically. The rescue path must build a FRESH, tool-history-free
    prompt and try once more — proving it actually avoids replaying the
    problematic history rather than just retrying the same thing.
    """
    TOOL_FAIL_ERROR = (
        "Error code: 400 - {'error': {'message': 'Tool choice is none, but "
        "model called a tool', 'code': 'tool_use_failed', "
        "'failed_generation': '{\"name\": \"calculator\", \"arguments\": "
        "{\"end_value\": 66987000000}}'}}"
    )

    seen_message_shapes: list[list] = []

    class FlakySynth:
        def __init__(self): self.n = 0
        def bind_tools(self, tools, **k): return self
        def invoke(self, messages):
            self.n += 1
            seen_message_shapes.append(list(messages))
            if self.n == 1:
                # The main synthesis attempt (tools bound, tool_choice=none).
                # groq_client._FallbackChatGroq's own internal retry (which
                # replays the same history and, per the real production
                # trace, fails identically) is tested separately in
                # test_model_fallback_*; this unit test stubs
                # invoke_with_retry directly to isolate chat_agent_node's
                # OWN rescue logic, so only one call happens before it.
                raise Exception(TOOL_FAIL_ERROR)
            # Second call is the RESCUE attempt from chat_agent.py itself —
            # succeeds because it's a fresh, minimal prompt.
            return AIMessage(content="MSFT FCF CAGR was -4.90% per the retrieved data.")

    fake = FlakySynth()
    import nodes.chat_agent as ca, nodes.intent_router as ir, nodes.summarizer as sm
    for module in (ca, ir, sm):
        module.get_llm = lambda *a, **k: fake
        module.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)

    preserved = [{
        "tool": "financial_data_api", "args": {"ticker": "MSFT"},
        "result": "Period: 2024-06-30\nFree Cash Flow: 74071000000\n"
                  "Period: 2026-06-30\nFree Cash Flow: 66987000000\n",
    }]

    state = {
        "messages": [HumanMessage(content="MSFT FCF CAGR?")],
        "user_input": "MSFT FCF CAGR?",
        "tool_loops": 4,  # budget exhausted -> final round -> tools bound w/ tool_choice=none
        "successful_tool_results": preserved,
        "tool_trace": [],
        "active_ticker": "MSFT",
    }

    result = ca.chat_agent_node(state)

    check("rescue synthesis succeeded (turn COMPLETE, not partial)",
          result.get("turn_status") == "COMPLETE", str(result.get("turn_status")))
    answer = str(result["messages"][0].content)
    check("rescued answer is a real synthesis, not the raw partial-response format",
          "Partial result" not in answer, answer[:200])
    check("rescued answer contains the actual figure", "-4.90%" in answer, answer)

    # The critical assertion: the RESCUE call's message list must NOT
    # contain the original tool-call-laden history — only the fresh,
    # condensed prompt. This is what makes it structurally different from
    # groq_client's own retry (which replays the same messages and, per the
    # real trace, failed identically because of that).
    rescue_call_messages = seen_message_shapes[-1]
    check("rescue call used a fresh 2-message prompt, not the full history",
          len(rescue_call_messages) == 2, f"got {len(rescue_call_messages)} messages")
    check("rescue prompt contains the retrieved data as plain text",
          "66987000000" in str(rescue_call_messages[-1].content),
          str(rescue_call_messages[-1].content)[:300])


def test_rescue_synthesis_does_not_fire_for_unrelated_errors():
    """A 429 or a genuine 500 must NOT trigger the rescue path — that would
    burn an extra LLM call on a problem the rescue can't fix, and could
    mask a real bug as if it were the tool_use_failed pattern."""
    class AlwaysRateLimited:
        def bind_tools(self, tools, **k): return self
        def invoke(self, messages):
            raise Exception("Error code: 429 - rate_limit_exceeded")

    fake = AlwaysRateLimited()
    import nodes.chat_agent as ca, nodes.intent_router as ir, nodes.summarizer as sm
    call_count = {"n": 0}

    def counting_invoke(llm, msgs, **k):
        call_count["n"] += 1
        return llm.invoke(msgs)

    for module in (ca, ir, sm):
        module.get_llm = lambda *a, **k: fake
        module.invoke_with_retry = counting_invoke

    state = {
        "messages": [HumanMessage(content="x")],
        "user_input": "x",
        "tool_loops": 0,
        "successful_tool_results": [{"tool": "t", "args": {}, "result": "data"}],
        "tool_trace": [],
    }
    result = ca.chat_agent_node(state)

    check("429 still falls through to partial response, not rescue",
          "Partial result" in str(result["messages"][0].content),
          str(result["messages"][0].content)[:200])
    check("no extra rescue LLM call was made for a 429",
          call_count["n"] == 1, f"invoke_with_retry called {call_count['n']} times")


def test_sec_xbrl_normalizer_parses_scaled_values():
    """
    SEC's XBRL text format is structurally different from yfinance's — it
    uses scaled human-readable values ('$391.04B') with the period embedded
    INLINE per line, not raw integers under a shared 'Period:' header. This
    proves the dedicated SEC parser handles that shape correctly.
    """
    from database.connection import init_db
    from database.normalizer import normalize_sec_xbrl_text, record_source
    from database.connection import query

    init_db(force=True)
    sample = (
        "[SEC XBRL] SECTEST 10-K FY2024\n"
        "  Total Revenue                                 :      $391.04B  (FY end: 2024-09-28)\n"
        "  Net Income                                    :       $93.74B  (FY end: 2024-09-28)\n"
        "  EPS (Diluted)                                 :        6.0800  (FY end: 2024-09-28)\n"
        "  Cash & Equivalents                            :       $29.94B  (FY end: 2024-09-28)\n"
        "  Long-Term Debt                                :       $96.70B  (FY end: 2024-09-28)\n"
        "  Shareholders Equity                           :       $56.95B  (FY end: 2024-09-28)\n"
    )
    src = record_source(provider="sec", source_type="filing")
    result = normalize_sec_xbrl_text("SECTEST", sample, src, period_type="annual")

    check("SEC normalizer wrote an income row", result["income"] == 1, str(result))
    check("SEC normalizer wrote a balance row", result["balance"] == 1, str(result))

    income = query(
        "SELECT revenue, net_income, eps_diluted FROM income_statements "
        "WHERE company_id IN (SELECT id FROM companies WHERE ticker='SECTEST')"
    )
    check("scaled $391.04B correctly parsed to raw float",
          abs(income[0]["revenue"] - 391_040_000_000) < 1, str(income[0] if income else None))
    check("unscaled EPS value parsed correctly", income[0]["eps_diluted"] == 6.08, str(income[0]))

    balance = query(
        "SELECT total_equity FROM balance_sheets "
        "WHERE company_id IN (SELECT id FROM companies WHERE ticker='SECTEST')"
    )
    check("SEC balance sheet label mapping worked ('Shareholders Equity' -> total_equity)",
          abs(balance[0]["total_equity"] - 56_950_000_000) < 1, str(balance[0] if balance else None))


def test_sec_normalizer_skips_na_lines_safely():
    """N/A lines (see the earlier honest-unavailable-year fix) must not
    crash the parser or produce garbage rows."""
    from database.normalizer import normalize_sec_xbrl_text, record_source

    src = record_source(provider="sec", source_type="filing")
    sample = "  Total Revenue                                 : N/A for FY1999 (most recent available: 2024-09-28)\n"
    result = normalize_sec_xbrl_text("NATEST", sample, src, period_type="annual")
    check("N/A lines produce zero rows, not a crash", result == {"income": 0, "balance": 0}, str(result))


def test_calculation_fidelity_catches_mismatched_answer():
    """
    Deterministic ground-truth check: if the calculator returned one number
    and the final answer states a DIFFERENT number for the same
    calculation, that's a real, verifiable fidelity failure — not an LLM
    opinion.
    """
    from evaluation.evaluator import check_calculation_fidelity

    successful_results = [{
        "tool": "calculator",
        "args": {"metric": "cagr"},
        "result": "[Calculator] CAGR = 16.35%",
    }]

    good_answer = "Microsoft's revenue CAGR was 16.35% over the period."
    result_good = check_calculation_fidelity(successful_results, good_answer)
    check("matching answer scores 100", result_good["score"] == 100, str(result_good))
    check("no mismatches recorded", result_good["mismatches"] == [], str(result_good))

    bad_answer = "Microsoft's revenue CAGR was 14.20% over the period."
    result_bad = check_calculation_fidelity(successful_results, bad_answer)
    check("mismatched answer scores 0", result_bad["score"] == 0, str(result_bad))
    check("mismatch is recorded with detail", len(result_bad["mismatches"]) == 1, str(result_bad))


def test_calculation_fidelity_not_applicable_when_no_calculator_used():
    """If the turn never called the calculator, the dimension is N/A (None),
    not a score — same convention as reconciliation_score with no
    contradictions. A None must not be mistaken for a 0 (bad) in the rollup."""
    from evaluation.evaluator import check_calculation_fidelity

    result = check_calculation_fidelity(
        [{"tool": "financial_data_api", "args": {}, "result": "Revenue: 100"}],
        "Revenue was $100.",
    )
    check("no calculator call -> not applicable (None)", result is None, str(result))


def test_thread_scoped_evaluation_query():
    """The thread_id column has existed since evaluator.py was built; this
    tests the query function that was missing for it."""
    from database.connection import execute, init_db
    import evaluation.evaluator as _evaluator
    from evaluation.evaluator import scores_for_thread
    from datetime import datetime as _dt, timezone as _tz
    import json as _json

    init_db(force=True)
    _evaluator._init()

    for turn_id, thread in [("th-a-1", "thread-alpha"), ("th-a-2", "thread-alpha"), ("th-b-1", "thread-beta")]:
        execute(
            """INSERT INTO evaluation_scores
               (thread_id, turn_id, query, answer, turn_status, overall_score,
                judge_summary, turn_metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (thread, turn_id, "q", "a", "COMPLETE", 80, "s",
             _json.dumps({}), _dt.now(_tz.utc).isoformat()),
        )

    alpha_scores = scores_for_thread("thread-alpha")
    check("thread-scoped query returns only that thread's turns",
          len(alpha_scores) == 2, str(len(alpha_scores)))
    check("does not leak turns from a different thread",
          all(t["turn_id"].startswith("th-a") for t in alpha_scores), str(alpha_scores))


def test_provenance_recorded_for_unstructured_content():
    """
    Web search / transcripts have no structured typed table to normalize
    into, but they should STILL get a sources row recorded — this is the
    fix to cached_call's gating logic (was: required BOTH normalize AND
    provider; now: provider alone is enough for provenance-only recording).
    """
    from database.connection import init_db, query
    from database.cache_service import cached_call

    init_db(force=True)
    before = query("SELECT COUNT(*) as n FROM sources WHERE provider = 'tavily'")[0]["n"]

    cached_call(
        "web_search", {"query": f"unique test query {os.getpid()}"}, "news",
        lambda: "[WebSearch] Results for: test",
        provider="tavily", source_type="web",
    )

    after = query("SELECT COUNT(*) as n FROM sources WHERE provider = 'tavily'")[0]["n"]
    check("a sources row was recorded for an unstructured (no-normalizer) call",
          after == before + 1, f"before={before} after={after}")


def test_postgres_insert_returns_real_id_not_zero():
    """
    Regression test for a real bug found via live-Postgres verification:
    psycopg2 cursors HAVE a .lastrowid attribute (so the old getattr default
    never kicked in), but it's always None/0 on Postgres — there is no
    session-level 'last insert id' the way SQLite has. The row was being
    inserted correctly; only the REPORTED id was wrong, which then silently
    became a bad foreign key elsewhere. Skips gracefully if no live Postgres
    is reachable — this is an integration test, not something that should
    block a SQLite-only environment.
    """
    import os as _os
    pg_url = _os.environ.get("TEST_DATABASE_URL", "postgresql://postgres:testpass@localhost:5432/finance_test")

    try:
        import psycopg2
        test_conn = psycopg2.connect(pg_url)
        test_conn.close()
    except Exception:
        print("  SKIP  no live Postgres reachable — this is expected outside the dev sandbox")
        return

    original_url = _os.environ.get("DATABASE_URL")
    _os.environ["DATABASE_URL"] = pg_url
    try:
        from database.connection import execute, init_db, query
        init_db(force=True)

        new_id = execute(
            "INSERT INTO companies (ticker, created_at, updated_at) VALUES (?, ?, ?)",
            (f"IDTEST{_os.getpid()}", "x", "x"),
        )
        check("execute() returns a real nonzero id on Postgres", new_id > 0, str(new_id))

        actual = query("SELECT id FROM companies WHERE ticker = ?", (f"IDTEST{_os.getpid()}",))
        check("returned id matches the actual stored row",
              actual and actual[0]["id"] == new_id, f"returned={new_id} actual={actual}")
    finally:
        if original_url is not None:
            _os.environ["DATABASE_URL"] = original_url
        else:
            _os.environ.pop("DATABASE_URL", None)


def test_eval_enabled_flag_parses_correctly():
    """The EVAL_ENABLED env flag gates whether the per-turn judge call fires
    at all — 'off' must mean zero Groq spend on evaluation. Verifies the
    truthy/falsy parsing so a typo in .env fails safe (defaults on) rather
    than silently disabling scoring."""
    import importlib
    import config as _config

    original = os.environ.get("EVAL_ENABLED")
    try:
        for value, expected in [
            (None, True),      # unset -> default on
            ("true", True),
            ("false", False),
            ("0", False),
            ("off", False),
            ("no", False),
            ("yes", True),     # anything not explicitly falsy -> on
        ]:
            if value is None:
                os.environ.pop("EVAL_ENABLED", None)
            else:
                os.environ["EVAL_ENABLED"] = value
            reloaded = importlib.reload(_config)
            check(f"EVAL_ENABLED={value!r} -> {expected}",
                  reloaded.EVAL_ENABLED == expected,
                  f"got {reloaded.EVAL_ENABLED}")
    finally:
        if original is not None:
            os.environ["EVAL_ENABLED"] = original
        else:
            os.environ.pop("EVAL_ENABLED", None)
        importlib.reload(_config)


def test_source_tier_grades_providers_correctly():
    """The tier assignments are the whole basis of the contradiction
    detector's 'lead with higher tier' rule. If they drift, the wrong
    source wins every dispute."""
    from utils.contradiction_detector import source_tier

    check("SEC is tier A", source_tier("sec") == "A")
    check("FRED is tier A", source_tier("fred") == "A")
    check("yfinance is tier B", source_tier("yfinance") == "B")
    check("Alpha Vantage is tier B", source_tier("alpha_vantage") == "B")
    check("Tavily/web is tier D", source_tier("tavily") == "D")
    check("unknown provider defaults to tier C", source_tier("mystery") == "C")


def test_contradiction_detector_finds_meaningful_disagreements():
    """
    The core Phase B guarantee: when two sources disagree on the same
    (company, period_end, field) beyond the noise threshold, the detector
    surfaces it — with a primary_recommendation pointing at the higher tier.
    """
    from database.connection import execute, init_db
    from database.normalizer import upsert_company, record_source
    from utils.contradiction_detector import find_contradictions_from_state

    # Force re-init: DB file may have been wiped between test runs while the
    # module-level "_INITIALISED = True" cache in database.connection stayed
    # in memory. Without this, upsert would hit "no such table: companies".
    init_db(force=True)

    company_id = upsert_company("CONTRADICT")

    # Two sources disagree on FY2024 net income: yfinance $88.1B, SEC $105B
    # — a ~16% delta that's clearly above the 5% noise threshold. The
    # original test used 88.1 vs 92.5, which came out at 4.76% — actually a
    # good demonstration of the threshold working, but not what this test
    # is checking. Numbers picked deliberately larger here so the test is
    # about the detection logic, not the exact threshold.
    sec_source = record_source(provider="sec", source_type="filing")
    execute(
        """INSERT INTO income_statements
           (company_id, period_end, period_type, net_income, source_id, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (company_id, "2024-06-30", "annual", 105000000000, sec_source,
         datetime.now(timezone.utc).isoformat()),
    )

    # Now the tool just returned yfinance's version
    fresh_payload = (
        "[FinData] Income Statement (Annual) – CONTRADICT\n"
        "Period: 2024-06-30\n"
        "  Total Revenue      : 245122000000\n"
        "  Net Income         : 88100000000\n"
    )
    state = {
        "successful_tool_results": [{
            "tool": "financial_data_api",
            "args": {"ticker": "CONTRADICT"},
            "result": fresh_payload,
            "status": "success",
        }],
        "active_ticker": "CONTRADICT",
    }

    contradictions = find_contradictions_from_state(state)
    check("contradiction was found", len(contradictions) >= 1, str(len(contradictions)))

    ni = next((c for c in contradictions if c["field"] == "net_income"), None)
    check("net_income disagreement recorded", ni is not None, str(contradictions))
    if ni:
        check("primary recommendation prefers SEC (tier A) over yfinance (tier B)",
              ni["primary_recommendation"]["provider"] == "sec",
              str(ni["primary_recommendation"]))


def test_contradiction_detector_ignores_noise():
    """A 0.1% rounding difference between yfinance and AV is not a
    contradiction — that would spam the synthesis prompt with useless
    reconciliation notes for every fact ever fetched."""
    from database.connection import execute, init_db
    from database.normalizer import upsert_company, record_source
    from utils.contradiction_detector import find_contradictions_from_state

    init_db(force=True)
    company_id = upsert_company("NOISE")
    src = record_source(provider="alpha_vantage", source_type="api")
    execute(
        """INSERT INTO income_statements
           (company_id, period_end, period_type, revenue, source_id, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (company_id, "2024-12-31", "annual", 100_000_500_000, src,
         datetime.now(timezone.utc).isoformat()),
    )

    fresh = (
        "[FinData] Income Statement (Annual) – NOISE\n"
        "Period: 2024-12-31\n"
        "  Total Revenue      : 100000000000\n"
    )
    state = {
        "successful_tool_results": [{
            "tool": "financial_data_api", "args": {"ticker": "NOISE"},
            "result": fresh, "status": "success",
        }],
        "active_ticker": "NOISE",
    }

    contradictions = find_contradictions_from_state(state)
    check("0.0005% delta is correctly ignored as noise",
          len(contradictions) == 0, str(contradictions))


def test_contradiction_detector_fails_safe():
    """A broken detector must NOT crash a turn — return [] on any error."""
    from utils.contradiction_detector import find_contradictions_from_state

    # Malformed state, obviously wrong shape
    result = find_contradictions_from_state({"successful_tool_results": "not a list"})
    check("bad input returns [], not exception", result == [], str(result))


def test_evaluator_persists_and_rolls_up():
    """
    End-to-end: score two synthetic turns, then check the rollup surfaces
    both AND flags the low-scoring dimension.
    """
    from database.connection import execute, init_db
    import json as _json
    import evaluation.evaluator as _evaluator
    from evaluation.evaluator import rollup_report
    from datetime import datetime as _dt, timezone as _tz

    init_db(force=True)
    # evaluator._init also runs at import time but its own module state has
    # no "force" flag; call it again explicitly so the evaluation_scores
    # table exists in the wiped DB. Idempotent — CREATE IF NOT EXISTS.
    _evaluator._init()

    # Bypass the LLM judge — insert rows directly. That's what the test is
    # here to verify: the storage + aggregation, not the judge's opinion.
    for turn_id, scores in [
        ("t-good", {"overall": 88, "grounding": 90, "citation": 85,
                    "reconciliation": None, "completeness": 90, "format": 85}),
        ("t-bad",  {"overall": 42, "grounding": 30, "citation": 40,
                    "reconciliation": 20, "completeness": 60, "format": 55}),
    ]:
        execute(
            """INSERT INTO evaluation_scores
               (thread_id, turn_id, query, answer, turn_status,
                grounding_score, citation_score, reconciliation_score,
                completeness_score, format_score, overall_score,
                judge_summary, turn_metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("t-test", turn_id, "test query", "test answer", "COMPLETE",
             scores["grounding"], scores["citation"], scores["reconciliation"],
             scores["completeness"], scores["format"], scores["overall"],
             f"summary for {turn_id}", _json.dumps({"tool_count": 1, "contradiction_count": 0}),
             _dt.now(_tz.utc).isoformat()),
        )

    report = rollup_report(days=1)
    check("rollup counted both turns", report["n_turns_scored"] >= 2,
          str(report.get("n_turns_scored")))
    check("recent_low_scoring surfaced t-bad",
          any(r["turn_id"] == "t-bad" for r in report["recent_low_scoring"]),
          str(report["recent_low_scoring"]))
    check("worst_dimension is surfaced when there's a real one",
          report["worst_dimension"] is not None, str(report["worst_dimension"]))


def test_evaluator_score_clamping():
    """The judge sometimes returns 150 or -5 or 'high' — clamping keeps the
    DB honest."""
    from evaluation.evaluator import _clamp_score

    check("integer stays as-is", _clamp_score(75) == 75)
    check("above 100 clamps to 100", _clamp_score(150) == 100)
    check("negative clamps to 0", _clamp_score(-10) == 0)
    check("None returns None", _clamp_score(None) is None)
    check("string non-numeric returns None", _clamp_score("high") is None)
    check("float rounds to int", _clamp_score(72.6) == 73)


def test_evaluator_parse_tolerates_code_fences():
    """Judge LLMs sometimes wrap JSON in ```json fences despite instructions."""
    from evaluation.evaluator import _parse_judge_response

    ok = '{"overall_score": 80, "summary": "fine"}'
    check("plain JSON parses", _parse_judge_response(ok)["overall_score"] == 80)

    fenced = '```json\n{"overall_score": 70, "summary": "with fence"}\n```'
    check("fenced JSON parses too",
          _parse_judge_response(fenced)["overall_score"] == 70)


def test_model_fallback_advances_past_a_deprecated_model():
    """
    Regression test for the exact production incident: Groq returned 404
    'model_not_found' for the primary model (llama-3.3-70b-versatile at the
    time, but the specific name doesn't matter — any deprecated/inaccessible
    model should behave the same way). Confirms get_llm()'s returned object
    actually advances to the next candidate on this error, instead of the
    old behavior where the "fallback chain" could never engage because the
    failure only ever surfaced one function away from where the fallback
    loop lived (see _FallbackChatGroq's docstring for the full story).
    """
    import utils.groq_client as gc

    call_log = []

    class FakeChatGroqInstance:
        def __init__(self, model):
            self.model = model
        def bind_tools(self, *a, **k):
            return self
        def invoke(self, messages):
            call_log.append(self.model)
            if self.model == gc._FALLBACK_MODELS[0]:
                raise Exception(
                    "Error code: 404 - {'error': {'message': 'The model "
                    f"`{self.model}` does not exist or you do not have "
                    "access to it.', 'code': 'model_not_found'}}"
                )
            return f"response from {self.model}"

    def fake_chatgroq_constructor(model, temperature, max_tokens):
        return FakeChatGroqInstance(model)

    original_chatgroq = gc.ChatGroq
    original_env = os.environ.pop("GROQ_MODEL", None)
    gc.ChatGroq = fake_chatgroq_constructor
    try:
        llm = gc.get_llm(temperature=0.1, max_tokens=100)
        result = llm.invoke([])
        check("first candidate was tried and failed", gc._FALLBACK_MODELS[0] in call_log,
              str(call_log))
        check("fallback advanced to the SECOND candidate", gc._FALLBACK_MODELS[1] in call_log,
              str(call_log))
        check("second candidate's response was returned",
              result == f"response from {gc._FALLBACK_MODELS[1]}", str(result))
    finally:
        gc.ChatGroq = original_chatgroq
        if original_env is not None:
            os.environ["GROQ_MODEL"] = original_env


def test_model_fallback_does_not_mask_unrelated_errors():
    """
    A 429 or a genuine 400 must NOT trigger model fallback — that would
    silently paper over real bugs (rate limits, malformed requests) by
    making them look like a model-availability problem, and would also
    burn through every candidate model for a problem none of them have.
    """
    import utils.groq_client as gc

    class AlwaysRateLimited:
        def __init__(self, model):
            self.model = model
        def bind_tools(self, *a, **k):
            return self
        def invoke(self, messages):
            raise Exception("Error code: 429 - rate_limit_exceeded. Please try again in 0.05s")

    original_chatgroq = gc.ChatGroq
    original_env = os.environ.pop("GROQ_MODEL", None)
    gc.ChatGroq = lambda model, temperature, max_tokens: AlwaysRateLimited(model)
    try:
        llm = gc.get_llm(temperature=0.1, max_tokens=100)
        try:
            llm.invoke([])
            check("a 429 must propagate, not be swallowed as model fallback", False,
                  "no exception raised")
        except Exception as exc:
            check("429 propagates as-is (not rewritten as 'all models unavailable')",
                  "429" in str(exc) and "All Groq models unavailable" not in str(exc),
                  str(exc))
    finally:
        gc.ChatGroq = original_chatgroq
        if original_env is not None:
            os.environ["GROQ_MODEL"] = original_env


def test_get_llm_failure_is_caught_not_just_invoke_failure():
    """
    Regression test for a real bug this file's own tests caught: get_llm()
    was called OUTSIDE the try/except in both chat_agent_node and
    summarizer_node, so a get_llm failure (e.g. missing GROQ_API_KEY, or any
    future validation added there) crashed the node instead of producing the
    same graceful "I hit an error" message that an invoke() failure gets.
    Simulates that specific failure point in both nodes.
    """
    import nodes.chat_agent as chat_agent_module
    import nodes.summarizer as summarizer_module

    def _boom(*a, **k):
        raise RuntimeError("simulated: GROQ_API_KEY missing")

    original_chat_get_llm = chat_agent_module.get_llm
    original_summarizer_get_llm = summarizer_module.get_llm
    chat_agent_module.get_llm = _boom
    summarizer_module.get_llm = _boom
    try:
        result = chat_agent_module.chat_agent_node(
            {"messages": [HumanMessage(content="hi")], "tool_loops": 0}
        )
        check("chat_agent_node survives a get_llm failure", "messages" in result, str(result))
        answer = str(result["messages"][0].content)
        check("chat_agent_node's error message is user-facing, not a stack trace",
              "error" in answer.lower(), answer)

        from nodes.summarizer import summarizer_node
        import config

        old_messages = []
        for i in range(config.CHAT_HISTORY_MAX_MESSAGES + 6):
            msg = HumanMessage(content=f"turn {i}")
            msg.id = f"g-{i}"
            old_messages.append(msg)
        summary_result = summarizer_node({"messages": old_messages})
        check("summarizer_node survives a get_llm failure", summary_result == {},
              str(summary_result))
    finally:
        chat_agent_module.get_llm = original_chat_get_llm
        summarizer_module.get_llm = original_summarizer_get_llm


def main() -> int:
    import nodes.chat_agent as chat_agent_module
    import nodes.intent_router as intent_router_module
    import nodes.summarizer as summarizer_module
    import utils.groq_client as groq_client_module

    # install_fake_llm() / install_fake_tools() monkeypatch module-level
    # functions and dicts and NEVER restored them — a real test-isolation bug
    # this file's own tests caught twice: first with TOOLS_BY_NAME leaking a
    # stubbed registry into a later test, then with a stubbed
    # invoke_with_retry leaking into test_daily_rate_limit_fails_fast, which
    # needs the REAL retry/backoff logic to actually test anything. `from
    # module import name` binds at CALL time, not once at file load, so a
    # test importing "the real function" can still receive whatever a prior
    # test patched the module attribute to. Snapshotting every mutable global
    # these helpers touch and restoring it after EVERY test — not just the
    # ones that call the install_* helpers — makes each test's environment
    # independent of run order, full stop.
    _snapshot = {
        "tools_by_name": dict(chat_agent_module.TOOLS_BY_NAME),
        "chat_get_llm": chat_agent_module.get_llm,
        "chat_invoke": chat_agent_module.invoke_with_retry,
        "intent_get_llm": intent_router_module.get_llm,
        "intent_invoke": intent_router_module.invoke_with_retry,
        "summarizer_get_llm": summarizer_module.get_llm,
        "summarizer_invoke": summarizer_module.invoke_with_retry,
        "groq_get_llm": groq_client_module.get_llm,
        "groq_invoke": groq_client_module.invoke_with_retry,
    }

    def _restore():
        chat_agent_module.TOOLS_BY_NAME = dict(_snapshot["tools_by_name"])
        chat_agent_module.get_llm = _snapshot["chat_get_llm"]
        chat_agent_module.invoke_with_retry = _snapshot["chat_invoke"]
        intent_router_module.get_llm = _snapshot["intent_get_llm"]
        intent_router_module.invoke_with_retry = _snapshot["intent_invoke"]
        summarizer_module.get_llm = _snapshot["summarizer_get_llm"]
        summarizer_module.invoke_with_retry = _snapshot["summarizer_invoke"]
        groq_client_module.get_llm = _snapshot["groq_get_llm"]
        groq_client_module.invoke_with_retry = _snapshot["groq_invoke"]

    tests = [
        test_smalltalk_skips_tools,
        test_tool_loop_executes_and_returns,
        test_loop_budget_forces_an_answer,
        test_report_requires_confirmation,
        test_bare_yes_without_pending_offer_is_not_a_report,
        test_memory_persists_across_turns,
        test_ticker_extraction_ignores_stopwords,
        test_history_window_never_orphans_tool_messages,
        test_tool_failure_is_surfaced_not_swallowed,
        test_report_type_override_by_typing,
        test_report_type_classifier_gets_ticker_on_fallback,
        test_token_windowing_trims_large_tool_results,
        test_gross_margin_accepts_gross_profit,
        test_registered_calculator_tool_accepts_gross_profit,
        test_summarizer_skips_short_history,
        test_summarizer_folds_old_messages_and_sets_summary,
        test_summarizer_merges_with_existing_summary,
        test_summarizer_failure_does_not_break_the_turn,
        test_chart_tool_trace_carries_full_untrimmed_args,
        test_chart_tool_rejects_mismatched_series_length,
        test_get_llm_failure_is_caught_not_just_invoke_failure,
        test_daily_rate_limit_fails_fast_without_sleeping,
        test_sec_xbrl_revenue_uses_modern_tag_not_stale_one,
        test_sec_xbrl_never_silently_substitutes_wrong_year,
        test_model_fallback_advances_past_a_deprecated_model,
        test_model_fallback_does_not_mask_unrelated_errors,
        test_synthesis_failure_preserves_fetched_data,
        test_synthesis_failure_with_no_data_gives_plain_error,
        test_preserved_results_reset_between_turns,
        test_formatter_scales_raw_figures,
        test_cache_prevents_duplicate_api_calls,
        test_cache_never_stores_error_responses,
        test_cache_failure_degrades_to_no_cache_not_crash,
        test_normalization_writes_typed_rows_with_provenance,
        test_db_rebuild_serves_different_shaped_requests_without_api,
        test_report_jobs_persist_across_restart,
        test_field_filter_trims_response_but_keeps_structure,
        test_field_filter_falls_back_safely_on_unknown_field,
        test_field_filter_no_fields_returns_full_payload,
        test_field_filter_handles_multiple_fields_and_periods,
        test_rescue_synthesis_recovers_when_retry_replays_bad_history,
        test_rescue_synthesis_does_not_fire_for_unrelated_errors,
        test_sec_xbrl_normalizer_parses_scaled_values,
        test_sec_normalizer_skips_na_lines_safely,
        test_calculation_fidelity_catches_mismatched_answer,
        test_calculation_fidelity_not_applicable_when_no_calculator_used,
        test_thread_scoped_evaluation_query,
        test_provenance_recorded_for_unstructured_content,
        test_postgres_insert_returns_real_id_not_zero,
        test_eval_enabled_flag_parses_correctly,
        test_source_tier_grades_providers_correctly,
        test_contradiction_detector_finds_meaningful_disagreements,
        test_contradiction_detector_ignores_noise,
        test_contradiction_detector_fails_safe,
        test_evaluator_persists_and_rolls_up,
        test_evaluator_score_clamping,
        test_evaluator_parse_tolerates_code_fences,
    ]

    for test in tests:
        print(f"\n{test.__name__}")
        try:
            test()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            FAILED.append(f"{test.__name__} (raised {type(exc).__name__})")
        finally:
            _restore()

    print(f"\n{'=' * 60}")
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"    - {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
