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

# The question-understanding step makes one extra model call per question. The
# end-to-end tests below were written, with scripted model replies, before it
# existed, and that extra call would consume those scripts. They run with it OFF
# (the rules-only router, exactly as before); the tests of the understanding step
# itself switch it on explicitly and restore it afterwards.
config.FRONT_DOOR_ENABLED = False

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


def test_transient_connection_error_retries_instead_of_failing_immediately():
    """
    Regression test for a real production trace: 4 financial_data_api calls
    succeeded, but the final synthesis call failed with a raw 'Connection
    error.' — a network-level failure, not a model or tool problem. The old
    code treated every non-429 error as permanent and raised immediately,
    throwing away 4 successful tool calls' worth of work straight to
    partial results. This verifies the fix: transient connection errors now
    get retried with short backoff before escalating.
    """
    from utils.groq_client import invoke_with_retry, _is_transient_connection_error

    check("classifier catches the exact production error",
          _is_transient_connection_error(Exception("Connection error.")), "should be True")
    check("classifier does not catch 429 (handled separately)",
          not _is_transient_connection_error(Exception("Error code: 429 - rate_limit_exceeded")),
          "should be False")
    check("classifier does not catch tool_use_failed (handled separately)",
          not _is_transient_connection_error(Exception("Tool choice is none, but model called a tool")),
          "should be False")

    class FlakyThenWorks:
        def __init__(self): self.n = 0
        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                raise Exception("Connection error.")
            return AIMessage(content="recovered after transient network failure")

    llm = FlakyThenWorks()
    import time as _time
    original_sleep = _time.sleep
    _time.sleep = lambda s: None   # don't actually wait in the test
    try:
        result = invoke_with_retry(llm, [HumanMessage(content="x")], max_retries=3)
    finally:
        _time.sleep = original_sleep

    check("recovers on the second attempt after a transient error",
          result.content == "recovered after transient network failure", result.content)
    check("actually retried (called invoke twice)", llm.n == 2, f"called {llm.n} times")


def test_transient_connection_error_eventually_gives_up():
    """A connection error that NEVER recovers must still raise after
    max_retries — not retry forever and hang the request."""
    from utils.groq_client import invoke_with_retry

    class AlwaysDown:
        def __init__(self): self.n = 0
        def invoke(self, messages):
            self.n += 1
            raise Exception("Connection error.")

    llm = AlwaysDown()
    import time as _time
    original_sleep = _time.sleep
    _time.sleep = lambda s: None
    try:
        raised = False
        try:
            invoke_with_retry(llm, [HumanMessage(content="x")], max_retries=3)
        except Exception:
            raised = True
    finally:
        _time.sleep = original_sleep

    check("gives up and raises after max_retries", raised, "should have raised")
    check("attempted exactly max_retries times", llm.n == 3, f"attempted {llm.n} times")


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


def test_calculator_arithmetic_operations():
    """
    Regression test for a real production failure: a user asked to add four
    quarters of GDP. The calculator had no addition operation — only
    financial formulas — so the model spiralled (1,361 reasoning tokens of
    'calculator with metric cagr? Not appropriate' repeated ~30 times) and
    emitted wrong CAGR calls. The system prompt promised 'use the calculator
    for all arithmetic' but the tool could not actually add. This verifies
    the arithmetic operations that close that gap.
    """
    from tools.calculator import calculator

    # The exact failing scenario: sum four quarters of NA000334Q GDP.
    # Correct answer is 29,384,172 million = $29.38T, which matches the
    # GDPA annual series — the number the model could not reach before.
    result = calculator(metric="sum", values=[7037663, 7336125, 7418435, 7591949])
    check("sum adds four quarters correctly",
          "29,384,172" in result, result)

    check("subtract works", "50" in calculator(metric="subtract", values=[100, 30, 20]),
          calculator(metric="subtract", values=[100, 30, 20]))
    check("multiply works", "40" in calculator(metric="multiply", values=[4, 5, 2]),
          calculator(metric="multiply", values=[4, 5, 2]))
    check("divide works", "25" in calculator(metric="divide", values=[100, 4]),
          calculator(metric="divide", values=[100, 4]))
    check("average works", "20" in calculator(metric="average", values=[10, 20, 30]),
          calculator(metric="average", values=[10, 20, 30]))
    check("percent_change is simple, not annualized",
          "20.00%" in calculator(metric="percent_change", start_value=100, end_value=120),
          calculator(metric="percent_change", start_value=100, end_value=120))


def test_calculator_arithmetic_error_handling():
    """Arithmetic ops must fail cleanly, not crash or return garbage."""
    from tools.calculator import calculator

    check("empty sum returns clear error",
          "Error" in calculator(metric="sum", values=[]),
          calculator(metric="sum", values=[]))
    check("division by zero caught",
          "zero" in calculator(metric="divide", values=[10, 0]).lower(),
          calculator(metric="divide", values=[10, 0]))
    check("subtract needs 2+ values",
          "Error" in calculator(metric="subtract", values=[5]),
          calculator(metric="subtract", values=[5]))


def test_overview_block_populates_companies_and_valuation_snapshots():
    """
    Found by auditing each tool's OUTPUT against the tables: the yfinance
    market-overview block is JSON, but the normalizer only parsed
    "Label : value" lines, so it never saw it. Result in production:
    companies.name/sector/industry were NULL for every ticker and
    valuation_snapshots had NO writer that ever ran (its only INSERT lived in
    normalize_av_overview, which nothing calls). upsert_company already took an
    `overview` dict with these exact key names — the call was never made.
    """
    import json as _json
    from database.connection import init_db, query
    from database.normalizer import normalize_yfinance_text, record_source

    init_db(force=True)
    overview = {
        "Company Name": "Overview Test Corp", "Sector": "Technology",
        "Industry": "Software", "Currency": "USD", "Current Price": 100.0,
        "Market Cap": 5e11, "Trailing P/E": 30.0, "Forward P/E": 25.0,
        "EV/EBITDA": 20.0, "Price/Book": 8.0, "Dividend Yield (%)": 0.5,
        "52-Week High": 120.0, "52-Week Low": 80.0, "50-Day MA": 98.0,
        "200-Day MA": 95.0, "Beta": 1.1, "Revenue (TTM)": 2e11,
        "Net Income (TTM)": 6e10, "Profit Margin (%)": 30.0,
        "Return on Equity (%)": 35.0, "Debt/Equity": 40.0,
    }
    text = ("[FinData] Market Overview \u2013 OVWTEST\n" + "=" * 60 + "\n"
            + _json.dumps(overview, indent=2))
    src = record_source(provider="yfinance", source_type="api")
    counts = normalize_yfinance_text("OVWTEST", text, src)

    check("a valuation snapshot is reported written", counts.get("valuation") == 1, str(counts))
    co = query("SELECT name, sector, industry, currency FROM companies WHERE ticker='OVWTEST'")[0]
    check("companies row now carries real metadata (was all NULL)",
          co["name"] == "Overview Test Corp" and co["sector"] == "Technology"
          and co["industry"] == "Software" and co["currency"] == "USD", str(dict(co)))

    v = query("SELECT * FROM valuation_snapshots WHERE company_id IN "
              "(SELECT id FROM companies WHERE ticker='OVWTEST')")[0]
    check("market cap and multiples stored",
          v["market_cap"] == 5e11 and v["pe"] == 30.0 and v["ev_ebitda"] == 20.0, str(dict(v)))
    check("price, 52-week range and moving averages stored",
          v["price"] == 100.0 and v["week52_high"] == 120.0 and v["ma_200"] == 95.0, str(dict(v)))
    check("profitability / leverage fields stored",
          v["roe"] == 35.0 and v["debt_to_equity"] == 40.0 and v["revenue_ttm"] == 2e11, str(dict(v)))

    overview["Current Price"] = 110.0
    text2 = ("[FinData] Market Overview \u2013 OVWTEST\n" + "=" * 60 + "\n"
             + _json.dumps(overview, indent=2))
    normalize_yfinance_text("OVWTEST", text2, src)
    rows = query("SELECT price FROM valuation_snapshots WHERE company_id IN "
                 "(SELECT id FROM companies WHERE ticker='OVWTEST')")
    check("second call the same day refreshes the row instead of duplicating it",
          len(rows) == 1 and rows[0]["price"] == 110.0, str([dict(r) for r in rows]))


def test_cashflow_dividends_and_buybacks_are_stored():
    """Both lines were printed on every cash-flow call but had no column, so
    the normalizer silently discarded them."""
    from database.connection import init_db, query
    from database.normalizer import normalize_yfinance_text, record_source

    init_db(force=True)
    text = (
        "[FinData] Cash Flow (Annual) \u2013 CFTEST\n"
        "\n  Period: 2025-06-30\n"
        "  Operating Cash Flow : 100\n  CapEx               : -30\n"
        "  Free Cash Flow      : 70\n  Dividends Paid      : -20\n"
        "  Share Buybacks      : -15\n"
    )
    src = record_source(provider="yfinance", source_type="api")
    normalize_yfinance_text("CFTEST", text, src)
    r = query("SELECT dividends_paid, share_repurchases, free_cash_flow FROM cash_flow_statements "
              "WHERE company_id IN (SELECT id FROM companies WHERE ticker='CFTEST')")[0]
    check("dividends_paid stored", r["dividends_paid"] == -20.0, str(dict(r)))
    check("share_repurchases stored", r["share_repurchases"] == -15.0, str(dict(r)))
    check("existing free_cash_flow unaffected", r["free_cash_flow"] == 70.0, str(dict(r)))


def test_schema_migration_upgrades_an_existing_database():
    """
    CREATE TABLE IF NOT EXISTS never alters a table that already exists, so
    editing a CREATE statement alone would leave every existing database file
    without the new columns. This builds an OLD-shaped table set and proves the
    migration adds the columns in place, keeps the data, and is safe to rerun.
    """
    import sqlite3
    from database.connection import _apply_migrations, _column_exists

    from database.schema import SCHEMA_MIGRATIONS

    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    # Build an "old" database from the migration list itself: one bare table
    # per table a migration touches, none of the new columns. Deriving this
    # (instead of hard-coding table names) means adding a migration can never
    # break this test again — twice it did, when a new table was added.
    for table in sorted({t for t, _, _ in SCHEMA_MIGRATIONS}):
        cur.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")
    cur.execute("ALTER TABLE macro_series ADD COLUMN series_code TEXT")
    cur.execute("INSERT INTO macro_series (series_code) VALUES ('GDP')")

    added = _apply_migrations(cur)
    check("migration reports the columns it added", added == len(SCHEMA_MIGRATIONS), f"added={added}")
    check("dividends_paid column now exists", _column_exists(cur, "cash_flow_statements", "dividends_paid"))
    check("valuation price column now exists", _column_exists(cur, "valuation_snapshots", "price"))
    check("seasonal_adjustment column now exists", _column_exists(cur, "macro_series", "seasonal_adjustment"))
    check("long_term_debt column now exists", _column_exists(cur, "balance_sheets", "long_term_debt"))
    check("existing row survives the migration",
          cur.execute("SELECT series_code FROM macro_series").fetchone()[0] == "GDP")
    check("running it a second time is a no-op", _apply_migrations(cur) == 0)


def test_fred_metadata_is_stored_and_shortcuts_resolve():
    """
    FRED's /series/observations (all this tool used to call) returns dates and
    values and nothing describing them, so macro_series.name was just the code
    repeated and unit/frequency stayed empty — and the agent could not tell a
    Seasonally Adjusted Annual Rate series from a raw quarterly one (the GDP
    confusion). The /series endpoint carries that; the tool now includes it.
    """
    from database.connection import init_db, query
    from database.normalizer import normalize_fred_series, record_source
    from tools.fred import resolve_series_id

    init_db(force=True)
    src = record_source(provider="fred", source_type="api")
    text = (
        "[FRED] Series: NA000334Q  (raw series ID)\n"
        "Title: Gross Domestic Product\n"
        "Units: Millions of Dollars | Frequency: Quarterly | Adjustment: Not Seasonally Adjusted\n"
        + "=" * 60 + "\n"
        "Date                    Value\n" + "-" * 30 + "\n"
        "2024-10-01       7,591,949.00\n"
    )
    written = normalize_fred_series("NA000334Q", "NA000334Q", text, src)
    check("observations still parsed with the new header present", written == 1, str(written))
    row = query("SELECT name, unit, frequency, seasonal_adjustment FROM macro_series "
                "WHERE series_code='NA000334Q'")[0]
    check("title replaces the placeholder name", row["name"] == "Gross Domestic Product", str(dict(row)))
    check("units, frequency and adjustment stored",
          row["unit"] == "Millions of Dollars" and row["frequency"] == "Quarterly"
          and row["seasonal_adjustment"] == "Not Seasonally Adjusted", str(dict(row)))

    plain = "[FRED] Series: GDPA  (raw series ID)\n" + "=" * 60 + "\nDate  Value\n" + "-" * 30 + "\n2024-01-01   29,380.99\n"
    normalize_fred_series("GDPA", "GDPA", plain, src)
    r2 = query("SELECT name, unit FROM macro_series WHERE series_code='GDPA'")[0]
    check("no metadata (endpoint down) -> row still created, name falls back to the code",
          r2["name"] == "GDPA" and r2["unit"] is None, str(dict(r2)))

    check("shortcut 'gdp' resolves to the real id (no duplicate series row)",
          resolve_series_id("gdp") == "GDP" and resolve_series_id("cpi") == "CPIAUCSL")


def test_dividend_yield_is_correct_under_either_yfinance_convention():
    """
    The tool used info["dividendYield"] * 100. yfinance changed that field from
    a fraction to a percent, so on a newer version it was multiplied twice and
    the agent reported Apple's yield as 31%. dividendRate / price does not
    depend on the version.
    """
    from tools.financial_data import _dividend_yield_pct as y

    as_percent = {"dividendRate": 1.04, "currentPrice": 260.0, "dividendYield": 0.4}
    as_fraction = {"dividendRate": 1.04, "currentPrice": 260.0, "dividendYield": 0.004}
    check("correct when yfinance reports dividendYield as a percent", y(as_percent) == 0.4, str(y(as_percent)))
    check("identical when yfinance reports it as a fraction", y(as_fraction) == 0.4, str(y(as_fraction)))
    check("non-dividend payer is 0, not an error", y({"currentPrice": 50.0}) == 0.0)
    check("zero price does not raise", y({"dividendRate": 1.0, "currentPrice": 0}) == 0.0)


def test_rescue_prompt_carries_context_contradictions_and_history():
    """
    The rescue used to receive only (question, data, ticker). So a follow-up
    like "add them all" lost what "them" meant, a flagged cross-source conflict
    never reached the model (while the UI banner still showed it), and the
    citation/tier rules were gone. It now builds its system prompt with the
    same _system_prompt() the normal path uses, in tools_available=False mode.
    """
    import nodes.chat_agent as ca

    captured = {}
    class LLM:
        def invoke(self, messages):
            captured["messages"] = list(messages)
            return AIMessage(content="rescued answer")
    saved = (ca.get_llm, ca.invoke_with_retry)
    ca.get_llm = lambda *a, **k: LLM()
    ca.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)
    try:
        out = ca._attempt_rescue_synthesis(
            user_question="add them all",
            preserved_results=[{"tool": "fred_series", "args": {"series_id": "NA000334Q"},
                                "result": "2024-01-01  7,037,663.00"}],
            active_ticker="MSFT",
            conversation_summary="SUMMARY-MARKER: user is asking about US GDP for 2024.",
            contradiction_note="NOTE-MARKER: SEC says 10, yfinance says 20.",
            history_text="User: what is the GDP of the US\nAssistant: here are the quarters",
        )
    finally:
        ca.get_llm, ca.invoke_with_retry = saved

    msgs = captured["messages"]
    system_text, human_text = str(msgs[0].content), str(msgs[1].content)
    check("rescue returns the model's answer", out == "rescued answer", str(out))
    check("still a fresh 2-message prompt (no tool-call history)", len(msgs) == 2, str(len(msgs)))
    check("no message carries tool_calls",
          not any(getattr(m, "tool_calls", None) for m in msgs))
    check("conversation summary reaches the rescue", "SUMMARY-MARKER" in system_text)
    check("contradiction note reaches the rescue", "NOTE-MARKER" in system_text)
    check("uses the tools-unavailable instruction", "tool budget" in system_text.lower(), system_text[-300:])
    check("earlier turns reach the rescue (so 'them' is resolvable)",
          "what is the GDP of the US" in human_text)
    check("retrieved data reaches the rescue", "7,037,663.00" in human_text)


def test_rescue_data_budget_is_shared_fairly_and_truncation_is_marked():
    """
    Before: each pass divided what was LEFT by the TOTAL count, so with five
    companies the first kept ~500 chars and the last ~205 — later companies
    were cut hardest, mid-table, with no sign anything was missing.
    """
    from nodes.chat_agent import _condense_for_rescue

    tickers = ["MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    recs = [{"tool": "financial_data_api", "args": {"ticker": t}, "result": "x" * 1000}
            for t in tickers]
    out = _condense_for_rescue(recs)
    sections = [sec for sec in out.split("\n\n") if sec.strip()]
    lengths = [len(sec) for sec in sections]
    check("every company is still present", len(sections) == 5, str(len(sections)))
    check("budget is shared evenly (spread <= 20 chars, was ~300)",
          max(lengths) - min(lengths) <= 20, str(lengths))
    check("cut data is explicitly marked", out.count("[truncated]") == 5, str(out.count("[truncated]")))

    short = {"tool": "t", "args": {}, "result": "short result"}
    longs = [{"tool": "t", "args": {"i": i}, "result": "y" * 3000} for i in range(2)]
    out2 = _condense_for_rescue([short] + longs)
    check("a short result keeps everything and is not marked",
          "short result" in out2 and out2.split("\n\n")[0].count("[truncated]") == 0)
    longest = max(len(sec) for sec in out2.split("\n\n"))
    check("budget a short result does not need flows to the long ones (>1,333 each)",
          longest > 1400, str(longest))

    small = _condense_for_rescue([{"tool": "t", "args": {}, "result": "tiny"}])
    check("nothing is marked when everything fits", "[truncated]" not in small)


def test_rescue_history_is_plain_text_without_tool_traffic():
    """History given to the rescue must not reintroduce the tool-call
    transcript the rescue exists to avoid."""
    from langchain_core.messages import ToolMessage
    from nodes.chat_agent import _plain_history

    msgs = [
        HumanMessage(content="what is the GDP of the US"),
        AIMessage(content="", tool_calls=[{"name": "fred_series", "args": {}, "id": "a"}]),
        ToolMessage(content="RAW-TOOL-OUTPUT", tool_call_id="a"),
        AIMessage(content="The US GDP is about 29.9 trillion."),
        HumanMessage(content="add them all"),                      # the CURRENT question
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "b"}]),
        ToolMessage(content="CURRENT-TURN-TOOL-RESULT", tool_call_id="b"),
    ]
    out = _plain_history(msgs)
    check("earlier question kept", "User: what is the GDP of the US" in out, out)
    check("earlier final answer kept", "Assistant: The US GDP is about 29.9 trillion." in out, out)
    check("current question excluded (it is passed separately)", "add them all" not in out, out)
    check("tool results excluded", "RAW-TOOL-OUTPUT" not in out and "CURRENT-TURN" not in out, out)


def test_rescued_turn_is_tagged_and_normal_turn_is_not():
    """Rescued answers run with narrower context, so evaluation needs to be
    able to tell them apart from normal ones (both are turn_status COMPLETE)."""
    import nodes.chat_agent as ca

    class Flaky:
        def __init__(self): self.n = 0
        def bind_tools(self, *a, **k): return self
        def invoke(self, messages):
            self.n += 1
            if self.n == 1:
                raise Exception("Error code: 400 - Tool choice is none, but model called a tool (tool_use_failed)")
            return AIMessage(content="rescued")
    fake = Flaky()
    saved = (ca.get_llm, ca.invoke_with_retry)
    ca.get_llm = lambda *a, **k: fake
    ca.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)
    try:
        state = {"messages": [HumanMessage(content="q")], "user_input": "q", "tool_loops": 4,
                 "successful_tool_results": [{"tool": "t", "args": {}, "result": "data"}],
                 "tool_trace": [], "active_ticker": None}
        rescued = ca.chat_agent_node(state)

        class Ok:
            def bind_tools(self, *a, **k): return self
            def invoke(self, messages): return AIMessage(content="normal answer")
        ca.get_llm = lambda *a, **k: Ok()
        normal = ca.chat_agent_node({**state, "tool_loops": 0, "successful_tool_results": []})
    finally:
        ca.get_llm, ca.invoke_with_retry = saved

    check("rescued turn is tagged 'rescue'", rescued.get("synthesis_mode") == "rescue", str(rescued.get("synthesis_mode")))
    check("normal turn carries no rescue tag", not normal.get("synthesis_mode"), str(normal.get("synthesis_mode")))


def test_evaluator_records_synthesis_mode_in_metadata():
    """The tag must reach the stored evaluation row, otherwise it is invisible
    to the rollup."""
    import json as _json
    import utils.groq_client as gc
    from database.connection import init_db, query
    import evaluation.evaluator as ev

    init_db(force=True)
    ev._init()

    class Resp:
        content = ('{"grounding_score": 90, "citation_score": 90, "reconciliation_score": null, '
                   '"completeness_score": 90, "format_score": 90, "overall_score": 90, "summary": "ok"}')
    class Judge:
        def invoke(self, m): return Resp()
    saved = (gc.get_llm, gc.invoke_with_retry)
    gc.get_llm = lambda *a, **k: Judge()
    gc.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)
    try:
        ev.score_turn(turn_id="mode-test-1", thread_id="mode-thread", question="q", answer="a",
                      successful_tool_results=[], contradictions=[], turn_status="COMPLETE",
                      synthesis_mode="rescue")
        ev.score_turn(turn_id="mode-test-2", thread_id="mode-thread", question="q", answer="a",
                      successful_tool_results=[], contradictions=[], turn_status="COMPLETE")
    finally:
        gc.get_llm, gc.invoke_with_retry = saved

    rows = {r["turn_id"]: _json.loads(r["turn_metadata"]) for r in
            query("SELECT turn_id, turn_metadata FROM evaluation_scores WHERE thread_id='mode-thread'")}
    check("rescued turn stored with synthesis_mode=rescue",
          rows.get("mode-test-1", {}).get("synthesis_mode") == "rescue", str(rows))
    check("default turn stored as normal",
          rows.get("mode-test-2", {}).get("synthesis_mode") == "normal", str(rows))


def test_statement_currency_is_tagged_and_stored():
    """
    yfinance returns each company in its OWN currency and never converts. The
    tool used to print bare numbers, so nothing — model or database — knew
    whether 2,400,000,000,000 was dollars or rupees, and the `currency` columns
    stayed empty. Statements now carry [reporting currency: XXX] and the
    normalizer stores it.
    """
    from database.connection import init_db, query
    from database.normalizer import normalize_yfinance_text, record_source

    init_db(force=True)
    src = record_source(provider="yfinance", source_type="api")
    text = (
        "[FinData] Income Statement (Annual) \u2013 TCSTEST.NS [reporting currency: INR]\n"
        "\n  Period: 2025-03-31\n  Total Revenue      : 2400000000000\n  Net Income         : 480000000000\n"
        "\n[FinData] Balance Sheet (Annual) \u2013 TCSTEST.NS [reporting currency: INR]\n"
        "\n  Period: 2025-03-31\n  Total Assets       : 1500000000000\n"
        "\n[FinData] Cash Flow (Annual) \u2013 TCSTEST.NS [reporting currency: INR]\n"
        "\n  Period: 2025-03-31\n  Operating Cash Flow : 500000000000\n"
    )
    normalize_yfinance_text("TCSTEST.NS", text, src)
    cid = "(SELECT id FROM companies WHERE ticker='TCSTEST.NS')"
    for table in ("income_statements", "balance_sheets", "cash_flow_statements"):
        ccy = query(f"SELECT currency FROM {table} WHERE company_id = {cid}")[0]["currency"]
        check(f"{table}.currency is stored (INR)", ccy == "INR", str(ccy))


def test_trading_and_reporting_currency_are_kept_separate():
    """An ADR trades in one currency and reports in another (e.g. Infosys: USD
    on a US exchange, INR statements). One column could not hold both facts."""
    import json as _json
    from database.connection import init_db, query
    from database.normalizer import normalize_yfinance_text, record_source

    init_db(force=True)
    src = record_source(provider="yfinance", source_type="api")
    overview = {"Company Name": "ADR Test Ltd", "Currency": "USD", "Reporting Currency": "INR",
                "Exchange": "NYQ", "Country": "India", "Market Cap": 8e10}
    text = ("[FinData] Market Overview \u2013 ADRTEST\n" + "=" * 60 + "\n" + _json.dumps(overview, indent=2)
            + "\n\n[FinData] Income Statement (Annual) \u2013 ADRTEST [reporting currency: INR]\n"
            "\n  Period: 2025-03-31\n  Total Revenue      : 1500000000000\n")
    normalize_yfinance_text("ADRTEST", text, src)
    co = query("SELECT currency, financial_currency, exchange, country FROM companies WHERE ticker='ADRTEST'")[0]
    check("trading currency kept (USD)", co["currency"] == "USD", str(dict(co)))
    check("reporting currency kept separately (INR)", co["financial_currency"] == "INR", str(dict(co)))
    check("exchange and country stored", co["exchange"] == "NYQ" and co["country"] == "India", str(dict(co)))
    inc = query("SELECT currency FROM income_statements WHERE company_id IN "
                "(SELECT id FROM companies WHERE ticker='ADRTEST')")[0]
    check("statement rows carry the REPORTING currency, not the trading one", inc["currency"] == "INR", str(dict(inc)))


def test_refresh_fills_currency_on_rows_that_already_exist():
    """
    Real bug found while adding this: the income upsert's ON CONFLICT ... DO
    UPDATE list left out currency (and EPS / cost of revenue). A row written
    before this change would therefore have kept a NULL currency forever, no
    matter how often it was refreshed.
    """
    from database.connection import init_db, query
    from database.normalizer import normalize_yfinance_text, record_source

    init_db(force=True)
    src = record_source(provider="yfinance", source_type="api")
    old_style = ("[FinData] Income Statement (Annual) \u2013 REFRESHTEST\n"
                 "\n  Period: 2025-06-30\n  Total Revenue      : 100\n  EPS (Basic)        : 1.5\n")
    normalize_yfinance_text("REFRESHTEST", old_style, src)
    row = query("SELECT currency, eps_basic FROM income_statements WHERE company_id IN "
                "(SELECT id FROM companies WHERE ticker='REFRESHTEST')")[0]
    check("a row written with no currency tag has NULL currency", row["currency"] is None, str(dict(row)))

    new_style = ("[FinData] Income Statement (Annual) \u2013 REFRESHTEST [reporting currency: USD]\n"
                 "\n  Period: 2025-06-30\n  Total Revenue      : 100\n  EPS (Basic)        : 1.9\n")
    normalize_yfinance_text("REFRESHTEST", new_style, src)
    rows = query("SELECT currency, eps_basic FROM income_statements WHERE company_id IN "
                 "(SELECT id FROM companies WHERE ticker='REFRESHTEST')")
    check("refresh did not duplicate the row", len(rows) == 1, str(len(rows)))
    check("refresh filled in the currency", rows[0]["currency"] == "USD", str(dict(rows[0])))
    check("refresh also updated EPS (was never in the UPDATE list)", rows[0]["eps_basic"] == 1.9, str(dict(rows[0])))


def test_sec_rows_are_tagged_usd():
    """tools/sec_filing.py reads only USD and USD-per-share units, so SEC rows
    are USD by construction."""
    from database.connection import init_db, query
    from database.normalizer import normalize_sec_xbrl_text, record_source

    init_db(force=True)
    src = record_source(provider="sec", source_type="filing")
    text = ("  Total Revenue                                 :      $391.04B  (FY end: 2024-09-28)\n"
            "  Shareholders Equity                           :       $56.95B  (FY end: 2024-09-28)\n")
    normalize_sec_xbrl_text("SECUSD", text, src, period_type="annual")
    cid = "(SELECT id FROM companies WHERE ticker='SECUSD')"
    check("SEC income row currency is USD",
          query(f"SELECT currency FROM income_statements WHERE company_id = {cid}")[0]["currency"] == "USD")
    check("SEC balance row currency is USD",
          query(f"SELECT currency FROM balance_sheets WHERE company_id = {cid}")[0]["currency"] == "USD")


def test_currency_lookup_separates_trading_from_reporting_and_does_not_cache_failures():
    import tools.financial_data as fd

    fd._CURRENCY_CACHE.clear()
    adr = {"currency": "USD", "financialCurrency": "INR"}
    check("ADR: trading USD, reporting INR", fd._currencies("ADRX", info=adr) == ("USD", "INR"))

    class Stock:
        def __init__(self, info): self._info = info
        @property
        def info(self): return self._info
    check("header tag uses the REPORTING currency",
          fd._ccy_tag("TAGX", Stock({"currency": "USD", "financialCurrency": "INR"})) == " [reporting currency: INR]")
    check("falls back to the trading currency when no reporting currency is given",
          fd._ccy_tag("TAGY", Stock({"currency": "EUR"})) == " [reporting currency: EUR]")

    class Boom:
        @property
        def info(self): raise RuntimeError("network down")
    fd._CURRENCY_CACHE.clear()
    check("a failed lookup yields no tag instead of raising", fd._ccy_tag("BOOMX", Boom()) == "")
    check("the failure was NOT cached (retried next time)",
          "BOOMX" not in fd._CURRENCY_CACHE, str(fd._CURRENCY_CACHE))
    check("Alpha Vantage's reportedCurrency is used for its statements",
          fd._av_ccy_tag([{"reportedCurrency": "INR"}]) == " [reporting currency: INR]")
    check("no AV reports -> no tag", fd._av_ccy_tag([]) == "")
    fd._CURRENCY_CACHE.clear()


def test_fx_rate_tool_direct_inverse_errors_and_caching():
    """
    The conversion tool returns a RATE; the model multiplies with the calculator
    so all arithmetic stays in one place. Network-free test: yfinance is faked.
    """
    import pandas as pd
    import tools as tools_pkg
    import tools.fx as fx
    from database.connection import init_db

    init_db(force=True)
    quotes = {"USDINR=X": 83.25, "INRGBP=X": 0.0095}      # GBPINR=X deliberately absent
    class FakeTicker:
        def __init__(self, symbol): self.symbol = symbol
        def history(self, period="5d"):
            if self.symbol in quotes:
                return pd.DataFrame({"Close": [quotes[self.symbol]]}, index=pd.to_datetime(["2026-10-24"]))
            return pd.DataFrame()
    saved = fx.yf.Ticker
    fx.yf.Ticker = FakeTicker
    try:
        direct = fx.fx_rate("usd", "inr")
        check("direct pair works and states the rate", "1 USD = 83.250000 INR" in direct, direct)
        check("states it is a market reference rate", "reference rate" in direct, direct)

        inverse = fx.fx_rate("GBP", "INR")      # only INRGBP=X exists -> inverted
        check("falls back to the inverse pair", "inverted from yfinance INRGBP=X" in inverse, inverse)
        check("inverted value is 1/0.0095", "105.263158" in inverse, inverse)

        check("same currency is a no-op", "same currency" in fx.fx_rate("INR", "INR"))
        check("bad code is rejected clearly", "Error:" in fx.fx_rate("RUPEES", "USD"))
        check("unknown pair reports no data", "Error:" in fx.fx_rate("ZZZ", "QQQ"))

        calls = {"n": 0}
        real = tools_pkg._fx
        def counting(**kw):
            calls["n"] += 1
            return real(**kw)
        tools_pkg._fx = counting
        try:
            tools_pkg.fx_rate_fn("USD", "INR"); tools_pkg.fx_rate_fn("usd", "inr")
        finally:
            tools_pkg._fx = real
        check("second identical request is served from cache", calls["n"] == 1, str(calls["n"]))

        fx.yf.Ticker = type("Dead", (), {"__init__": lambda self, s: None,
                                         "history": lambda self, period="5d": (_ for _ in ()).throw(ConnectionError("down"))})
        check("a network failure is reported, not raised", "Error:" in fx.fx_rate("EUR", "JPY"))
    finally:
        fx.yf.Ticker = saved


def test_fx_tool_is_chat_only_and_prompt_teaches_currency_and_suffix_rules():
    """The user asked that conversion exist only when requested, and that the
    report pipeline stay untouched."""
    import tools as tools_pkg
    import nodes.chat_agent as ca

    check("fx_rate is registered", "fx_rate" in tools_pkg.TOOLS_BY_NAME)
    check("chat can call fx_rate", "fx_rate" in [t.name for t in ca.CHAT_TOOLS])
    check("report fetch executor does NOT get fx_rate", "fx_rate" not in tools_pkg.FETCH_TOOL_FUNCTIONS)
    check("report planner is NOT told about fx_rate", "fx_rate" not in tools_pkg.TOOL_DESCRIPTIONS)

    # Collapse whitespace: the rule is wrapped across lines in the prompt source.
    prompt = " ".join(ca._system_prompt(None, tools_available=True).split())
    check("prompt forbids silently comparing across currencies", "Never add, compare or rank" in prompt)
    check("prompt says convert only on request", "Convert ONLY when the user explicitly asks" in prompt)
    check("prompt teaches the exchange suffixes", ".NS" in prompt and ".BO" in prompt)

    doc = tools_pkg.TOOLS_BY_NAME["financial_data_api"].description
    check("the ticker parameter the model reads explains .NS / .BO and the US default",
          ".NS" in doc and ".BO" in doc and "bare ticker" in doc.lower(), doc[:300])


def test_exact_text_cache_is_checked_before_the_db_rebuild():
    """
    Layer order: (1) exact-match text cache, (2) typed-table rebuild, (3) live.
    The old order put the rebuild first, so an IDENTICAL repeat was answered with
    a thinner reconstruction even though the full original was sitting in the
    cache. The text cache can never be worse (richer when both hit, a free miss
    otherwise), so it goes first; the DB still serves what the cache cannot — a
    differently-shaped request for the same data.
    """
    import tools as tools_pkg
    from database.connection import init_db

    init_db(force=True)
    calls = {"n": 0}
    def fake_findata(**kw):
        calls["n"] += 1
        return ("[FinData] Income Statement (Annual) \u2013 ORDX [reporting currency: USD]\n"
                "\n  Period: 2025-06-30\n  Total Revenue      : 300\n  Net Income         : 10\n"
                "\n  Period: 2024-06-30\n  Total Revenue      : 250\n  Net Income         : 9\n"
                "\nORIGINAL-ONLY-MARKER (a field the typed tables do not store)\n")
    real = tools_pkg._findata
    tools_pkg._findata = fake_findata
    try:
        first = tools_pkg.financial_data_api_fn("ORDX", "income", "annual", 2)
        second = tools_pkg.financial_data_api_fn("ORDX", "income", "annual", 2)
        shaped = tools_pkg.financial_data_api_fn("ORDX", "income", "annual", 1)
    finally:
        tools_pkg._findata = real

    check("first call went live", calls["n"] == 1, str(calls["n"]))
    check("identical repeat returns the ORIGINAL text, not a rebuild",
          second == first and "ORIGINAL-ONLY-MARKER" in second and "[FinData:DB]" not in second,
          second[:120])
    check("differently-shaped request (1 year vs 2) is rebuilt from the DB",
          shaped.startswith("[FinData:DB]"), shaped[:80])
    check("neither repeat caused another live call", calls["n"] == 1, str(calls["n"]))


def test_db_rebuild_keeps_currency_tag_and_cashflow_fields():
    """The rebuilt text must look like the live text. It was dropping the
    reporting-currency tag and the Dividends Paid / Share Buybacks lines."""
    from database.connection import init_db
    from database.normalizer import normalize_yfinance_text, record_source
    from database.financial_reads import try_rebuild_financials

    init_db(force=True)
    live = (
        "[FinData] Cash Flow (Annual) \u2013 RBLD.NS [reporting currency: INR]\n"
        "\n  Period: 2025-03-31\n  Operating Cash Flow : 500\n  CapEx               : -60\n"
        "  Free Cash Flow      : 440\n  Dividends Paid      : -200\n  Share Buybacks      : -150\n"
        "\n  Period: 2024-03-31\n  Operating Cash Flow : 470\n  CapEx               : -55\n"
        "  Free Cash Flow      : 415\n  Dividends Paid      : -180\n  Share Buybacks      : -10\n"
        "\n[FinData] Balance Sheet (Annual) \u2013 RBLD.NS [reporting currency: INR]\n"
        "\n  Period: 2025-03-31\n  Total Assets       : 900\n"
        "\n  Period: 2024-03-31\n  Total Assets       : 800\n")
    normalize_yfinance_text("RBLD.NS", live, record_source("yfinance", "api"))

    cf = try_rebuild_financials("RBLD.NS", "cashflow", "annual", 2) or ""
    check("cash-flow rebuild carries the reporting currency", "[reporting currency: INR]" in cf, cf[:90])
    check("Dividends Paid survives the round trip", "Dividends Paid" in cf and "-200" in cf, cf)
    check("Share Buybacks survives the round trip", "Share Buybacks" in cf and "-150" in cf, cf)
    bal = try_rebuild_financials("RBLD.NS", "balance", "annual", 2) or ""
    check("balance-sheet rebuild carries the reporting currency", "[reporting currency: INR]" in bal, bal[:90])


def test_all_is_never_served_as_a_partial_db_rebuild():
    """statement_type='all' (the tool's DEFAULT) is five blocks live. The typed
    tables can rebuild three, so serving them silently dropped the Market
    Overview and Key Ratios while looking complete."""
    from database.connection import init_db
    from database.normalizer import normalize_yfinance_text, record_source
    from database.financial_reads import try_rebuild_financials

    init_db(force=True)
    def blk(kind, body):
        return (f"\n[FinData] {kind} (Annual) \u2013 ALLX\n"
                f"\n  Period: 2025-06-30\n{body}\n  Period: 2024-06-30\n{body}")
    live = (blk("Income Statement", "  Total Revenue      : 1\n") + blk("Balance Sheet", "  Total Assets       : 1\n")
            + blk("Cash Flow", "  Operating Cash Flow : 1\n"))
    normalize_yfinance_text("ALLX", live, record_source("yfinance", "api"))

    check("every individual statement IS rebuildable",
          all(try_rebuild_financials("ALLX", k, "annual", 2) for k in ("income", "balance", "cashflow")))
    check("'all' is a miss (falls through to cache/live) instead of a partial answer",
          try_rebuild_financials("ALLX", "all", "annual", 2) is None)


def test_refetch_replaces_rows_and_never_duplicates_a_fiscal_year():
    """
    Real bug: record_source() creates a new sources row per retrieval and the
    fact tables are unique on (..., source_id), so the upsert never fired across
    real refetches. Every refetch added a second copy of every period, and the
    rebuild — LIMIT counting rows, not years — returned the latest year twice
    and dropped the other. Also: a SEC row (different provider, period_end a day
    off) must coexist for contradiction detection but never leak into the
    yfinance rebuild.
    """
    from database.connection import init_db, query
    from database.normalizer import (normalize_yfinance_text, normalize_sec_xbrl_text,
                                     record_source)
    from database.financial_reads import try_rebuild_financials

    init_db(force=True)
    body = ("[FinData] Income Statement (Annual) \u2013 DUPX\n"
            "\n  Period: 2025-06-30\n  Total Revenue      : {a}\n  Net Income         : 10\n"
            "\n  Period: 2024-06-30\n  Total Revenue      : {b}\n  Net Income         : 9\n")
    normalize_yfinance_text("DUPX", body.format(a=300, b=250), record_source("yfinance", "api"))
    normalize_yfinance_text("DUPX", body.format(a=301, b=250), record_source("yfinance", "api"))   # TTL expired, refetched
    normalize_yfinance_text("DUPX", body.format(a=302, b=250), record_source("yfinance", "api"))

    cid = "(SELECT id FROM companies WHERE ticker='DUPX')"
    rows = query(f"SELECT period_end, revenue FROM income_statements WHERE company_id = {cid} ORDER BY period_end DESC")
    check("three fetches still leave exactly one row per fiscal year",
          [r["period_end"] for r in rows] == ["2025-06-30", "2024-06-30"], str([dict(r) for r in rows]))
    check("the newest fetch won", rows[0]["revenue"] == 302.0, str([dict(r) for r in rows]))

    sec = ("  Total Revenue                                 :      $0.31B  (FY end: 2025-06-29)\n")
    normalize_sec_xbrl_text("DUPX", sec, record_source("sec", "filing"), period_type="annual")
    check("the SEC row coexists (different provider) for contradiction detection",
          query(f"SELECT COUNT(*) c FROM income_statements WHERE company_id = {cid}")[0]["c"] == 3)

    out = try_rebuild_financials("DUPX", "income", "annual", 2) or ""
    check("rebuild returns TWO distinct fiscal years", out.count("Period:") == 2
          and "2025-06-30" in out and "2024-06-30" in out, out)
    check("the SEC period never leaks into the yfinance-shaped rebuild", "2025-06-29" not in out, out)


def test_legacy_duplicates_are_ignored_by_reads_and_removed_by_migration():
    """Databases written before the fix already hold duplicate rows. Reads must
    cope immediately; the idempotent cleanup must remove them without touching
    other providers or rows that have no source."""
    from database.connection import execute, init_db, query
    from database.normalizer import record_source
    from database.financial_reads import try_rebuild_financials
    from database.schema import DATA_MIGRATIONS

    init_db(force=True)
    execute("INSERT INTO companies (ticker, created_at, updated_at) VALUES ('LEGX', 'x', 'x')")
    cid = query("SELECT id FROM companies WHERE ticker='LEGX'")[0]["id"]
    yf1, yf2, yf3 = (record_source("yfinance", "api") for _ in range(3))
    sec = record_source("sec", "filing")
    now = "2099-01-01T00:00:00+00:00"
    def ins(period, rev, src, updated=now):
        execute("INSERT INTO income_statements (company_id, period_end, period_type, revenue, source_id, updated_at) "
                "VALUES (?, ?, 'annual', ?, ?, ?)", (cid, period, rev, src, updated))
    ins("2025-06-30", 1.0, yf1); ins("2025-06-30", 2.0, yf2); ins("2025-06-30", 3.0, yf3)   # three old fetches
    ins("2024-06-30", 9.0, yf1)
    ins("2025-06-30", 99.0, sec)                  # other provider — must survive
    ins("2025-06-30", 77.0, None)                 # no source at all — must survive

    out = try_rebuild_financials("LEGX", "income", "annual", 2) or ""
    check("read side: two distinct years even with legacy duplicates",
          out.count("Period:") == 2 and "2024-06-30" in out, out)

    for _ in range(2):                            # idempotent
        for stmt in DATA_MIGRATIONS:
            execute(stmt)
    left = query("SELECT revenue FROM income_statements WHERE company_id=? AND period_end='2025-06-30' ORDER BY revenue", (cid,))
    check("cleanup keeps the newest yfinance row, the SEC row and the source-less row",
          [r["revenue"] for r in left] == [3.0, 77.0, 99.0], str([dict(r) for r in left]))
    check("the older fiscal year is untouched",
          query("SELECT COUNT(*) c FROM income_statements WHERE company_id=? AND period_end='2024-06-30'", (cid,))[0]["c"] == 1)


class _ScriptedTool:
    """Returns (or raises) a scripted outcome per call; counts executions."""
    def __init__(self, name, outcomes):
        self.name, self.outcomes, self.calls = name, list(outcomes), 0
    def invoke(self, args):
        out = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(out, Exception):
            raise out
        return out


def _chat_run_one(tool, call_args=None, **kw):
    """Runs the real chat _run_one with the pause patched out; returns (record, sleeps)."""
    import nodes.chat_agent as ca
    sleeps = []
    saved = (ca.TOOLS_BY_NAME, ca.time.sleep)
    ca.TOOLS_BY_NAME = {tool.name: tool}
    ca.time.sleep = lambda secs: sleeps.append(secs)
    try:
        rec = ca._run_one({"name": tool.name, "args": call_args or {"q": 1}, "id": "c1"}, **kw)
    finally:
        ca.TOOLS_BY_NAME, ca.time.sleep = saved
    return rec, sleeps


def test_classifier_trusts_failure_keywords_only_on_error_shaped_text():
    """
    A successful web search ABOUT unauthorized trading, or an outage that "timed
    out", used to be labelled systemic because the failure patterns were searched
    across the whole result. Once SYSTEMIC disables a tool for the turn, that
    mislabel would withdraw a working tool and discard good results.
    """
    from utils.error_classifier import classify_result, SYSTEMIC, TRANSIENT, OK
    news = ("[WebSearch] Results for: 'Acme Corp news'\n" + "=" * 60 + "\n\n[Result 1]\n"
            "  Snippet : Regulators opened an inquiry into unauthorized trading by a former desk "
            "head after the exchange's feed timed out repeatedly. " * 4)
    check("long successful news mentioning 'unauthorized' / 'timed out' is OK",
          classify_result("web_search", news) == OK, classify_result("web_search", news))
    check("a short genuine auth error is still systemic",
          classify_result("t", "[SEC] Error: unauthorized") == SYSTEMIC)
    long_error = "[Tool] Error: " + "request details " * 40 + "Unauthorized - invalid API key"
    check("a LONG message with the explicit '[Tool] Error' prefix is still trusted",
          classify_result("t", long_error) == SYSTEMIC)
    check("an exception class name counts as transient",
          classify_result("t", "[t] Error: raised ConnectionError: Max retries exceeded") == TRANSIENT)


def test_chat_transient_failure_is_retried_in_code_after_a_pause():
    """TRANSIENT needs no reasoning: one retry, same arguments, after a pause —
    without spending a model call or a round."""
    import config
    flaky = _ScriptedTool("av", ["[AV] Error: Connection error: request timed out", "[AV] data ok"])
    rec, sleeps = _chat_run_one(flaky)
    check("the retry succeeded", rec["status"] == "success", str(rec))
    check("tool ran twice, retries recorded", flaky.calls == 2 and rec["retries"] == 1, f"{flaky.calls} {rec['retries']}")
    check("it paused before retrying", sleeps == [config.TRANSIENT_RETRY_PAUSE_S], str(sleeps))

    dead = _ScriptedTool("av", ["[AV] Error: HTTP 503 service unavailable"])
    rec2, _ = _chat_run_one(dead)
    check("a persistent transient failure stops after ONE retry (no loop)", dead.calls == 2, str(dead.calls))
    check("and is reported as a failed, transient call", rec2["status"] == "failed" and rec2["error_class"] == "transient", str(rec2))


def test_chat_systemic_failure_withdraws_the_tool_for_the_turn():
    """SYSTEMIC is never retried, and the tool is withdrawn for the rest of the
    turn: not offered to the model, and refused without executing if asked."""
    import nodes.chat_agent as ca

    tool = _ScriptedTool("sec_filing_search", ["[SEC] Error: Unauthorized - invalid API key"])
    rec, sleeps = _chat_run_one(tool)
    check("never retried", tool.calls == 1 and sleeps == [], f"calls={tool.calls} sleeps={sleeps}")
    check("classified systemic", rec["error_class"] == "systemic" and rec["status"] == "failed", str(rec))

    saved = ca.TOOLS_BY_NAME
    ca.TOOLS_BY_NAME = {"sec_filing_search": tool}
    try:
        node_out = ca.chat_tool_node({
            "messages": [AIMessage(content="", tool_calls=[{"name": "sec_filing_search", "args": {"q": 1}, "id": "n1"}])],
            "tool_loops": 0, "tool_trace": [],
        })
    finally:
        ca.TOOLS_BY_NAME = saved
    check("the node reports it as withdrawn", node_out["disabled_tools"] == ["sec_filing_search"], str(node_out["disabled_tools"]))

    executions_before = tool.calls
    refused, _ = _chat_run_one(tool, disabled=frozenset({"sec_filing_search"}))
    check("a withdrawn tool is refused WITHOUT executing",
          tool.calls == executions_before, f"before={executions_before} after={tool.calls}")
    check("the refusal tells the model what to do instead",
          "Do not call it again" in refused["content"] and refused["error_class"] == "systemic", refused["content"])

    seen = {}
    class LLM:
        def invoke(self, m): return AIMessage(content="ok")
    sg, si = ca.get_llm, ca.invoke_with_retry
    ca.get_llm = lambda *a, **k: (seen.update(tools=[t.name for t in (k.get("with_tools") or [])]), LLM())[1]
    ca.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)
    try:
        ca.chat_agent_node({"messages": [HumanMessage(content="q")], "user_input": "q", "tool_loops": 0,
                            "successful_tool_results": [], "tool_trace": [],
                            "disabled_tools": ["sec_filing_search"]})
    finally:
        ca.get_llm, ca.invoke_with_retry = sg, si
    check("the withdrawn tool is no longer bound for the model", "sec_filing_search" not in seen["tools"], str(seen))
    check("the other tools still are", "financial_data_api" in seen["tools"], str(seen))


def test_chat_bad_args_are_not_blindly_retried_and_repeats_are_refused():
    """BAD_ARGS needs the arguments to change, which only the model can do. The
    same arguments can only reproduce the error, so neither the code nor a
    repeated model request re-runs them."""
    import nodes.chat_agent as ca
    calc = _ScriptedTool("calculator", ["[Calculator] Error: start_value and years must be positive."])
    rec, sleeps = _chat_run_one(calc, {"metric": "cagr", "years": 0})
    check("executed once, no pause, no automatic retry", calc.calls == 1 and sleeps == [], f"{calc.calls} {sleeps}")
    check("classified bad_args and the model sees the actual error",
          rec["error_class"] == "bad_args" and "must be positive" in rec["content"], str(rec))

    key = ca._call_key("calculator", {"metric": "cagr", "years": 0})
    again, _ = _chat_run_one(calc, {"metric": "cagr", "years": 0}, failed_calls=frozenset({key}))
    check("an identical repeat is refused without executing", calc.calls == 1, str(calc.calls))
    check("and the refusal says to change the arguments", "already failed" in again["content"], again["content"])

    changed, _ = _chat_run_one(calc, {"metric": "cagr", "years": 2}, failed_calls=frozenset({key}))
    check("CHANGED arguments are allowed through", calc.calls == 2, str(calc.calls))


def test_chat_empty_result_tells_the_model_to_stop():
    tool = _ScriptedTool("sec_filing_search", ["No 10-K filings found for ZZZZ"])
    rec, _ = _chat_run_one(tool)
    check("an empty result is not a failure", rec["status"] == "empty", str(rec))
    check("never retried", tool.calls == 1)
    check("the model is told it is a valid final answer", "Do not retry" in rec["content"], rec["content"])


def test_chat_raised_exceptions_are_classified_from_their_text():
    """Every raised exception used to be labelled systemic — so a plain TypeError
    (wrong arguments) or a dropped connection looked like a permanently broken tool."""
    typo = _ScriptedTool("t", [TypeError("got an unexpected keyword argument 'bogus'")])
    rec, _ = _chat_run_one(typo)
    check("a TypeError for bad arguments is bad_args, not systemic",
          rec["error_class"] == "bad_args" and rec["status"] == "failed", str(rec))

    net = _ScriptedTool("t", [ConnectionError("Connection refused"), "[T] ok"])
    rec2, sleeps = _chat_run_one(net)
    check("a dropped connection is transient and retried", rec2["status"] == "success" and net.calls == 2, str(rec2))


def test_withdrawn_tools_reset_on_the_next_user_message():
    """End to end through the real graph: a tool withdrawn in one turn must get a
    fresh chance on the user's next message."""
    tool_call = lambda i, t: {"name": "sec_filing_search", "args": {"ticker": t}, "id": i}
    install_fake_llm([
        AIMessage(content="", tool_calls=[tool_call("a1", "ZZZZ")]), AIMessage(content="Filing source is down."),
        AIMessage(content="", tool_calls=[tool_call("b1", "YYYY")]), AIMessage(content="Still down."),
    ])
    fakes = install_fake_tools({"sec_filing_search": "[SEC] Error: Unauthorized - invalid API key"})
    graph = fresh_graph()

    v1 = run(graph, "get me ZZZZ's 10-K", thread="reset1")
    check("turn 1 withdrew the tool", v1.get("disabled_tools") == ["sec_filing_search"], str(v1.get("disabled_tools")))

    v2 = run(graph, "get me YYYY's 10-K", thread="reset1")
    check("turn 2 really executed the tool again (not refused)", fakes["sec_filing_search"].call_count == 2,
          str(fakes["sec_filing_search"].call_count))
    check("the withdrawal list was reset, then re-filled by turn 2's own failure",
          v2.get("disabled_tools") == ["sec_filing_search"], str(v2.get("disabled_tools")))


def test_report_fetch_transient_retry_waits_first():
    import config
    import nodes.fetch_executor as fe
    sleeps, calls = [], []
    outcomes = ["[AV] Error: Connection error: request timed out", "[AV] data ok"]
    saved = (fe.FETCH_TOOL_FUNCTIONS, fe.time.sleep)
    fe.FETCH_TOOL_FUNCTIONS = {"av": lambda **kw: (calls.append(kw), outcomes[len(calls) - 1])[1]}
    fe.time.sleep = lambda s_: sleeps.append(s_)
    try:
        r = fe._run_one_fetch_step({"tool": "av", "args": {"q": 1}, "step": 1}, set())
    finally:
        fe.FETCH_TOOL_FUNCTIONS, fe.time.sleep = saved
    check("transient failure recovered by one retry", r["status"] == "success" and len(calls) == 2, str(r))
    check("with the SAME arguments", calls[0] == calls[1])
    check("after a pause (it used to retry instantly)", sleeps == [config.TRANSIENT_RETRY_PAUSE_S], str(sleeps))
    check("retries_used counts the one retry", r["retries_used"] == 1, str(r["retries_used"]))


def test_report_fetch_bad_args_get_one_model_corrected_retry():
    """The fetch step used to retry bad arguments with the SAME arguments —
    guaranteed to fail the same way. It now asks the model to repair them once."""
    import nodes.fetch_executor as fe
    calls = []
    def tool(**kw):
        calls.append(kw)
        return "[FinData] Error: unexpected keyword argument 'bogus'" if "bogus" in kw else "[FinData] ok data"
    saved = (fe.FETCH_TOOL_FUNCTIONS, fe._correct_args)
    fe.FETCH_TOOL_FUNCTIONS = {"fd": tool}
    fe._correct_args = lambda name, args, err: {"ticker": "X"}
    try:
        r = fe._run_one_fetch_step({"tool": "fd", "args": {"ticker": "X", "bogus": 1}, "step": 1}, set())
    finally:
        fe.FETCH_TOOL_FUNCTIONS, fe._correct_args = saved
    check("repaired arguments succeed", r["status"] == "success", str(r))
    check("the retry used the CORRECTED arguments", calls == [{"ticker": "X", "bogus": 1}, {"ticker": "X"}], str(calls))

    calls.clear()
    fe.FETCH_TOOL_FUNCTIONS = {"fd": tool}
    fe._correct_args = lambda name, args, err: None
    try:
        r2 = fe._run_one_fetch_step({"tool": "fd", "args": {"ticker": "X", "bogus": 1}, "step": 2}, set())
    finally:
        fe.FETCH_TOOL_FUNCTIONS, fe._correct_args = saved
    check("no usable correction -> stop, no pointless identical retry", len(calls) == 1 and r2["status"] == "failed", f"{len(calls)} {r2}")
    check("failure is classified bad_args", r2["error_category"] == "bad_args", str(r2["error_category"]))


def test_report_fetch_systemic_and_empty_are_never_retried():
    import nodes.fetch_executor as fe
    sleeps, fixes = [], []
    saved = (fe.FETCH_TOOL_FUNCTIONS, fe.time.sleep, fe._correct_args)
    fe.time.sleep = lambda s_: sleeps.append(s_)
    fe._correct_args = lambda *a: fixes.append(a)
    try:
        n = {"sys": 0, "empty": 0}
        def sys_tool(**kw): n["sys"] += 1; return "[SEC] Error: Unauthorized - invalid API key"
        def empty_tool(**kw): n["empty"] += 1; return "No 10-K filings found for ZZZZ"
        fe.FETCH_TOOL_FUNCTIONS = {"sys": sys_tool, "empty": empty_tool}
        r1 = fe._run_one_fetch_step({"tool": "sys", "args": {}, "step": 1}, set())
        r2 = fe._run_one_fetch_step({"tool": "empty", "args": {}, "step": 2}, set())
    finally:
        fe.FETCH_TOOL_FUNCTIONS, fe.time.sleep, fe._correct_args = saved
    check("systemic: one attempt, flagged for disable + replan", n["sys"] == 1 and r1["status"] == "systemic_failure", str(r1))
    check("empty: one attempt, treated as a result, not a failure", n["empty"] == 1 and r2["status"] == "success", str(r2))
    check("neither paused nor asked the model to repair arguments", sleeps == [] and fixes == [], f"{sleeps} {fixes}")


def test_correct_args_helper_only_spends_a_model_call_when_it_can_help():
    import nodes.fetch_executor as fe

    asked = []
    class R:
        def __init__(self, c): self.content = c
    def install(reply):
        asked.clear()
        fe.get_llm = lambda *a, **k: object()
        fe.invoke_with_retry = lambda llm, msgs, **k: (asked.append(1), R(reply))[1]
    sg, si = fe.get_llm, fe.invoke_with_retry
    try:
        install('```json\n{"ticker": "X"}\n```')
        check("repaired arguments are parsed from a fenced reply",
              fe._correct_args("t", {"ticker": "X", "bogus": 1}, "unexpected keyword argument 'bogus'") == {"ticker": "X"})
        install('{"ticker": "X"}')
        check("identical arguments are rejected (nothing to retry)",
              fe._correct_args("t", {"ticker": "X"}, "invalid ticker") is None)
        install("{}")
        check("an empty object means 'cannot be fixed'", fe._correct_args("t", {"a": 1}, "invalid value") is None)
        install("not json at all")
        check("unparseable reply is safe", fe._correct_args("t", {"a": 1}, "invalid value") is None)
        install('{"ticker": "X"}')
        fe._correct_args("t", {"a": 1}, "KeyError: 'revenue'")
        check("no LLM call when the error does not read like an argument problem", asked == [], str(asked))
        fe.invoke_with_retry = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("llm down"))
        check("a failing model call never raises", fe._correct_args("t", {"a": 1}, "invalid value") is None)
    finally:
        fe.get_llm, fe.invoke_with_retry = sg, si


def _run_threads(*targets):
    import threading
    threads = [threading.Thread(target=t) for t in targets]
    for t in threads: t.start()
    for t in threads: t.join(timeout=10)
    return threads


def test_identical_concurrent_requests_share_one_live_call():
    """
    Cache stampede: N identical requests arriving together all miss before any
    has stored its result, so all N hit the live API (8 calls measured before
    this fix; 1 after). With a provider capped at 25 calls a day that is real
    money. The first request leads, the rest wait for its result.
    """
    import threading, time, os
    import database.cache_service as cs
    from database.connection import init_db
    init_db(force=True)
    calls = {"n": 0}; lock = threading.Lock(); out = []
    def fetch():
        with lock: calls["n"] += 1
        time.sleep(0.3)
        return "[FinData] Total Revenue : 100 (live)"
    args = {"ticker": f"COAL{os.getpid()}"}
    def worker(): out.append(cs.cached_call("financial_data_api", args, "annual_financials", fetch))
    started = time.time()
    _run_threads(*[worker] * 8)
    elapsed = time.time() - started

    check("eight identical simultaneous requests made ONE live call", calls["n"] == 1, str(calls["n"]))
    check("every caller received the data", len(out) == 8 and all(p == "[FinData] Total Revenue : 100 (live)" for p, _ in out), str(out))
    check("exactly one caller made the live call; seven were served without one",
          sorted(f for _, f in out) == [False] + [True] * 7, str(sorted(f for _, f in out)))
    check("waiting costs no extra latency (~one fetch, not eight)", elapsed < 0.8, f"{elapsed:.2f}s")
    check("the in-flight table is empty afterwards", cs._INFLIGHT == {}, str(cs._INFLIGHT))


def test_different_requests_are_not_serialized_by_coalescing():
    import threading, time, os
    import database.cache_service as cs
    from database.connection import init_db
    init_db(force=True)
    calls = {"n": 0}; lock = threading.Lock()
    def fetch():
        with lock: calls["n"] += 1
        time.sleep(0.3)
        return "[FinData] ok data"
    started = time.time()
    _run_threads(
        lambda: cs.cached_call("financial_data_api", {"ticker": f"DIFA{os.getpid()}"}, "annual_financials", fetch),
        lambda: cs.cached_call("financial_data_api", {"ticker": f"DIFB{os.getpid()}"}, "annual_financials", fetch),
    )
    check("two different requests each made their own call", calls["n"] == 2, str(calls["n"]))
    check("and ran in parallel, not one after the other", time.time() - started < 0.55, f"{time.time() - started:.2f}s")


def test_a_failing_leader_never_strands_or_poisons_the_waiters():
    """If the leading request raises there is no result to share. The waiter must
    not hang and must not inherit the failure; it should try the fetch itself."""
    import threading, time, os
    import database.cache_service as cs
    from database.connection import init_db
    init_db(force=True)
    calls = {"n": 0}; lock = threading.Lock(); outcome = {}
    def fetch():
        with lock:
            calls["n"] += 1; mine = calls["n"]
        if mine == 1:
            time.sleep(0.25)
            raise RuntimeError("provider exploded")
        return "[FinData] recovered data"
    args = {"ticker": f"FAIL{os.getpid()}"}
    def leader():
        try: cs.cached_call("financial_data_api", args, "annual_financials", fetch)
        except Exception as exc: outcome["leader"] = str(exc)
    def follower():
        time.sleep(0.05)
        outcome["follower"] = cs.cached_call("financial_data_api", args, "annual_financials", fetch)
    threads = _run_threads(leader, follower)
    check("nobody hung", not any(t.is_alive() for t in threads))
    check("the leader saw its own failure", outcome.get("leader") == "provider exploded", str(outcome))
    check("the waiter fetched for itself and succeeded", outcome.get("follower") == ("[FinData] recovered data", False), str(outcome))
    check("two live attempts in total (the failed one and the retry)", calls["n"] == 2, str(calls["n"]))
    check("in-flight table cleaned up after a failure", cs._INFLIGHT == {}, str(cs._INFLIGHT))


def test_an_error_result_is_shared_not_retried_by_every_waiter():
    """An error is not cached, but eight waiters each re-calling a provider that
    just returned an error (e.g. a rate limit) would hammer it. They share the
    leader's answer instead."""
    import threading, time, os
    import database.cache_service as cs
    from database.connection import init_db
    init_db(force=True)
    calls = {"n": 0}; lock = threading.Lock(); out = []
    def fetch():
        with lock: calls["n"] += 1
        time.sleep(0.2)
        return "[AV] Error: rate limit exceeded"
    args = {"ticker": f"ERRS{os.getpid()}"}
    def worker(): out.append(cs.cached_call("financial_data_api", args, "annual_financials", fetch))
    _run_threads(*[worker] * 4)
    check("four requests, one call to the failing provider", calls["n"] == 1, str(calls["n"]))
    check("all four got the same error text", {p for p, _ in out} == {"[AV] Error: rate limit exceeded"}, str(out))
    check("the error was not cached", cs.get_cached(cs.make_cache_key("financial_data_api", args)) is None)


def test_a_waiter_gives_up_on_a_stuck_leader():
    import threading, time, os
    import database.cache_service as cs
    from database.connection import init_db
    init_db(force=True)
    saved = cs._COALESCE_WAIT_S
    cs._COALESCE_WAIT_S = 0.05
    calls = {"n": 0}; lock = threading.Lock(); outcome = {}
    def fetch():
        with lock: calls["n"] += 1
        time.sleep(0.4)
        return "[FinData] slow data"
    args = {"ticker": f"SLOW{os.getpid()}"}
    try:
        threads = _run_threads(
            lambda: cs.cached_call("financial_data_api", args, "annual_financials", fetch),
            lambda: (time.sleep(0.05), outcome.update(f=cs.cached_call("financial_data_api", args, "annual_financials", fetch))),
        )
    finally:
        cs._COALESCE_WAIT_S = saved
    check("no thread hung behind the slow leader", not any(t.is_alive() for t in threads))
    check("the waiter fetched independently after the timeout", calls["n"] == 2, str(calls["n"]))
    check("and still got an answer", outcome.get("f", (None,))[0] == "[FinData] slow data", str(outcome))


def test_rate_limit_wait_is_not_spent_after_the_last_attempt():
    """After the 5th failed attempt the retry loop used to sleep once more
    (5-65 s) and only then report failure — a wait with nothing left to wait for."""
    import time as _time
    import utils.groq_client as gc
    tpm = ("Error code: 429 - {'error': {'message': 'Rate limit reached on tokens per minute (TPM): "
           "Please try again in 3.2s.', 'code': 'rate_limit_exceeded'}}")
    class LLM:
        def invoke(self, m): raise Exception(tpm)
    sleeps = []
    saved = _time.sleep; _time.sleep = lambda secs: sleeps.append(secs)
    try:
        err = None
        try: gc.invoke_with_retry(LLM(), ["hi"], max_retries=5)
        except Exception as exc: err = str(exc)
    finally:
        _time.sleep = saved
    check("5 attempts -> only 4 waits (none after the last)", len(sleeps) == 4, str(sleeps))
    check("it still reports exhaustion", err is not None and "All 5 retries exhausted" in err, str(err))


def test_daily_limit_falls_back_to_another_model_but_per_minute_limit_waits():
    """
    Groq limits are per model. A DAILY 429 on one model cannot be waited out, but
    another model has its own quota — so move on instead of failing. A PER-MINUTE
    429 clears in seconds and is still handled by waiting (invoke_with_retry), not
    by silently switching to a different model.
    """
    import utils.groq_client as gc
    tpd = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `m1` on tokens per day "
           "(TPD): Limit 200000', 'code': 'rate_limit_exceeded'}}")
    tpm = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `m1` on tokens per minute "
           "(TPM): Please try again in 3s.', 'code': 'rate_limit_exceeded'}}")
    behaviour, log = {}, []
    class FakeGroq:
        def __init__(self, model, temperature=0, max_tokens=0): self.model = model
        def invoke(self, messages):
            log.append(self.model)
            out = behaviour[self.model]
            if isinstance(out, Exception): raise out
            return out
    saved = gc.ChatGroq; gc.ChatGroq = FakeGroq
    try:
        def make(): return gc._FallbackChatGroq(["m1", "m2"], 0.1, 100, None, None)

        gc._DAILY_EXHAUSTED.clear(); log.clear()
        behaviour.update(m1=Exception(tpd), m2="answer-from-m2")
        check("daily limit on m1 -> answered by m2", make().invoke(["q"]) == "answer-from-m2", str(log))

        log.clear()
        check("the next request skips the exhausted model entirely", make().invoke(["q"]) == "answer-from-m2" and log == ["m2"], str(log))

        gc._DAILY_EXHAUSTED.clear(); log.clear()
        behaviour.update(m1=Exception(tpm), m2="answer-from-m2")
        raised = None
        try: make().invoke(["q"])
        except Exception as exc: raised = str(exc)
        check("a per-MINUTE limit is NOT a reason to switch models (it is waited out one layer up)",
              raised is not None and "tokens per minute" in raised and log == ["m1"], f"{raised} {log}")

        gc._DAILY_EXHAUSTED.clear(); log.clear()
        behaviour.update(m1=Exception(tpd), m2=Exception(tpd.replace("m1", "m2")))
        msg = None
        try: gc.invoke_with_retry(make(), ["q"])
        except Exception as exc: msg = str(exc)
        check("every model out of daily quota -> the clear 'daily quota exhausted' message",
              msg is not None and "Daily token quota exhausted" in msg, str(msg))
    finally:
        gc.ChatGroq = saved
        gc._DAILY_EXHAUSTED.clear()


def test_report_section_prompt_contains_the_users_original_request():
    """
    state['raw_query'] is documented as 'the question exactly as the user typed
    it', and the section instructions refer to 'the query' — but the writer was
    given only the ticker, the data and a slice of the outline. Scope words like
    a named fiscal year, 'in rupees' or 'valuation only' reached it only if the
    planning steps happened to carry them.
    """
    import nodes.report_generator as rg
    captured = {}
    class R: content = "section text"
    saved = rg.invoke_with_retry
    rg.invoke_with_retry = lambda llm, msgs, **k: (captured.update(msgs=msgs), R())[1]
    try:
        rg._write_section(object(), "EXEC_SUMMARY", "Executive Summary", "ctx", "TCS.NS", "{}",
                          user_query="Compare TCS revenue in rupees for FY2024 only")
        with_query = str(captured["msgs"][1].content)
        rg._write_section(object(), "EXEC_SUMMARY", "Executive Summary", "ctx", "TCS.NS", "{}")
        without = str(captured["msgs"][1].content)
    finally:
        rg.invoke_with_retry = saved
    check("the user's own words reach every section prompt",
          "Compare TCS revenue in rupees for FY2024 only" in with_query, with_query[:200])
    check("labelled so the model treats it as scope to respect", "ORIGINAL REQUEST" in with_query)
    check("callers that do not pass it still work", "N/A" in without)


def test_fair_share_and_the_judge_sees_every_tool_result_evenly():
    """The evaluation judge had the same unfair-truncation bug as the old rescue:
    later tool results were cut hardest, so the judge could call a correct figure
    in the fifth company 'ungrounded' while seeing only a sliver of its data."""
    from utils.text_budget import fair_share
    import evaluation.evaluator as ev

    blocks = ["a" * 1000, "b" * 1000, "c" * 1000, "d" * 1000, "e" * 1000]
    cut = fair_share(blocks, 2500)
    check("all blocks share the budget evenly (spread <= 20 chars)", max(map(len, cut)) - min(map(len, cut)) <= 20, str(list(map(len, cut))))
    check("anything cut is marked", all("[truncated]" in c for c in cut))
    mixed = fair_share(["tiny", "x" * 3000, "y" * 3000], 2000)
    check("a short block keeps everything and is not marked", mixed[0] == "tiny")
    check("its unused budget flows to the long blocks", len(mixed[1]) > 900 and len(mixed[2]) > 900, str(list(map(len, mixed))))
    check("under budget -> untouched", fair_share(["abc", "def"], 100) == ["abc", "def"])

    results = [{"tool": "financial_data_api", "args": {"ticker": t}, "result": "z" * 1000}
               for t in ("MSFT", "GOOGL", "AMZN", "META", "NVDA")]
    text = ev._summarise_tool_results(results, max_chars=2000)
    parts = [x for x in text.split("\n\n") if x.strip()]
    check("judge input keeps all five companies", len(parts) == 5, str(len(parts)))
    check("the last company gets as much as the first (was ~40% as much)",
          len(parts[-1]) >= 0.9 * len(parts[0]), f"{len(parts[0])} vs {len(parts[-1])}")


def _scripted_dispatch(front_replies, agent_replies, other=None):
    """
    A fake model that answers by WHICH step is asking, read from its system prompt:
    the understanding step, the out-of-scope reply, or the agent. Patches the shared
    client (the understanding step imports get_llm from it at call time) and every
    node/graph module. Returns (calls, restore) — calls records what each step saw.
    """
    import sys
    import utils.groq_client as gc
    calls = {"front": [], "agent": [], "scope": []}

    class LLM:
        def bind_tools(self, *a, **k): return self
        def invoke(self, messages):
            head = str(messages[0].content)
            if head.startswith("You read ONE message"):
                calls["front"].append(messages)
                reply = front_replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return AIMessage(content=reply)
            if "outside finance" in head:
                calls["scope"].append(messages)
                return AIMessage(content="I focus on financial research. Ask me about a company.")
            calls["agent"].append(messages)
            return agent_replies.pop(0)
    fake = LLM()
    saved = [(gc, "get_llm", gc.get_llm), (gc, "invoke_with_retry", gc.invoke_with_retry)]
    gc.get_llm = lambda *a, **k: fake
    gc.invoke_with_retry = lambda llm, m, **k: llm.invoke(m)
    for name, mod in list(sys.modules.items()):
        if name.startswith(("nodes.", "graph.")) and hasattr(mod, "get_llm"):
            saved.append((mod, "get_llm", mod.get_llm))
            mod.get_llm = lambda *a, **k: fake
            if hasattr(mod, "invoke_with_retry"):
                saved.append((mod, "invoke_with_retry", mod.invoke_with_retry))
                mod.invoke_with_retry = lambda llm, m, **k: llm.invoke(m)

    def restore():
        for mod, attr, original in reversed(saved):
            setattr(mod, attr, original)
    return calls, restore


def _brief_json(**kw):
    import json as _json
    base = {"in_scope": True, "intent": "chat", "entities": [], "metrics": [], "period": "latest",
            "uses_previous_context": False, "missing": [], "ask": None, "assumptions": []}
    base.update(kw)
    return _json.dumps(base)


_TICKS8 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX"]
_NAMES8 = ["Apple", "Microsoft", "Google", "Amazon", "Meta", "Nvidia", "Tesla", "Netflix"]


def _batch_state(args, tool_calls_id="call_1"):
    return {"messages": [HumanMessage(content="q"),
                         AIMessage(content="", tool_calls=[{"name": "financial_data_api", "args": args, "id": tool_calls_id}])],
            "tool_loops": 0, "tool_trace": [], "successful_tool_results": [], "disabled_tools": []}


class _TimedTool:
    def __init__(self, delay=0.0, outcomes=None):
        self.delay, self.outcomes, self.seen = delay, outcomes or {}, []
    def invoke(self, args):
        import time
        self.seen.append(args["ticker"])
        time.sleep(self.delay)
        return self.outcomes.get(args["ticker"], f"Market Cap for {args['ticker']}: 123")


def test_normalize_tickers_handles_lists_strings_names_and_the_cap():
    from tools import normalize_tickers
    kept, skipped = normalize_tickers("AAPL", ["msft", "AAPL", "Apple Inc", "nvda, tsla", "GC=F"])
    check("upper-cased, de-duplicated, order kept; a comma string is split",
          kept == ["AAPL", "MSFT", "NVDA", "TSLA", "GC=F"], str(kept))
    check("a company NAME is reported as not fetched instead of becoming fake tickers APPLE and INC",
          skipped == ["APPLE INC"], str(skipped))
    kept2, skipped2 = normalize_tickers(tickers="AAPL MSFT GOOGL AMZN META NVDA TSLA NFLX AMD ORCL")
    check("a space-separated string is split and capped at 8", len(kept2) == 8, str(kept2))
    check("the overflow is reported, not silently dropped", skipped2 == ["AMD", "ORCL"], str(skipped2))


def test_one_batch_call_is_one_round_with_one_reply_and_per_company_records():
    """
    In a real run the model made one call per round for eight companies, hit the
    4-round budget after four, and the rest came back 'not fetched'. A `tickers`
    list makes ONE call carry all of them; the tool node fans it out in parallel.
    """
    import time
    import nodes.chat_agent as ca
    tool = _TimedTool(delay=0.3)
    saved = ca.TOOLS_BY_NAME
    ca.TOOLS_BY_NAME = {"financial_data_api": tool}
    try:
        started = time.time()
        out = ca.chat_tool_node(_batch_state({"tickers": _TICKS8, "fields": ["market_cap", "pe"]}))
        elapsed = time.time() - started
    finally:
        ca.TOOLS_BY_NAME = saved
    check("one model call -> exactly ONE round used", out["tool_loops"] == 1, str(out["tool_loops"]))
    check("all eight companies were fetched", sorted(tool.seen) == sorted(_TICKS8), str(tool.seen))
    check("in parallel: ~one fetch of wall-clock, not eight (8 x 0.3 s sequential = 2.4 s)", elapsed < 1.2, f"{elapsed:.2f}s")
    check("the model gets exactly one ToolMessage, for the call it made",
          len(out["messages"]) == 1 and out["messages"][0].tool_call_id == "call_1", str(out["messages"]))
    check("the reply carries a header per company", all(f"=== {t} ===" in out["messages"][0].content for t in _TICKS8))
    check("one trace row per company (contradiction detection and the UI see each)", len(out["tool_trace"]) == 8, str(len(out["tool_trace"])))
    check("one preserved result per company, each with its own ticker arg",
          sorted(r["args"]["ticker"] for r in out["successful_tool_results"]) == sorted(_TICKS8))
    check("the shared `fields` argument reached every company",
          all(r["args"].get("fields") == ["market_cap", "pe"] for r in out["successful_tool_results"]))


def test_one_failing_company_does_not_fail_the_batch_and_is_named():
    import nodes.chat_agent as ca
    tool = _TimedTool(outcomes={"TSLA": "[FinData] Error: Connection error: request timed out"})
    saved = (ca.TOOLS_BY_NAME, ca.time.sleep)
    ca.TOOLS_BY_NAME = {"financial_data_api": tool}
    ca.time.sleep = lambda s_: None
    try:
        out = ca.chat_tool_node(_batch_state({"tickers": ["AAPL", "TSLA", "MSFT"]}))
    finally:
        ca.TOOLS_BY_NAME, ca.time.sleep = saved
    text = out["messages"][0].content
    check("the summary line names the failure", "2 ok" in text and "1 failed or empty (TSLA)" in text, text[:140])
    check("the failure text sits under its own company header", "=== TSLA ===\n[FinData] Error" in text)
    check("the others are unaffected", "Market Cap for AAPL" in text and "Market Cap for MSFT" in text)
    check("only the successes are preserved", sorted(r["args"]["ticker"] for r in out["successful_tool_results"]) == ["AAPL", "MSFT"])
    check("the transient failure was retried ONCE per company policy (TSLA ran twice)", tool.seen.count("TSLA") == 2, str(tool.seen))


def test_batch_cap_and_unrecognised_names_are_reported():
    import nodes.chat_agent as ca
    tool = _TimedTool()
    saved = ca.TOOLS_BY_NAME
    ca.TOOLS_BY_NAME = {"financial_data_api": tool}
    try:
        out = ca.chat_tool_node(_batch_state({"tickers": _TICKS8 + ["AMD", "ORCL", "Tata Motors"]}))
    finally:
        ca.TOOLS_BY_NAME = saved
    text = out["messages"][0].content
    check("only the first eight were fetched", len(tool.seen) == 8 and "AMD" not in tool.seen, str(tool.seen))
    check("the rest are listed as not fetched, with the reason",
          "Not fetched" in text and "AMD" in text and "ORCL" in text and "TATA MOTORS" in text, text[-220:])


def test_single_ticker_calls_are_untouched_and_a_packed_string_is_a_batch():
    import nodes.chat_agent as ca
    tool = _TimedTool()
    saved = ca.TOOLS_BY_NAME
    ca.TOOLS_BY_NAME = {"financial_data_api": tool}
    try:
        single = ca.chat_tool_node(_batch_state({"ticker": "MSFT"}))
        tool.seen.clear()
        packed = ca.chat_tool_node(_batch_state({"ticker": "AAPL, NVDA"}))
    finally:
        ca.TOOLS_BY_NAME = saved
    check("a plain single-ticker call returns the plain result, no batch wrapper",
          single["messages"][0].content == "Market Cap for MSFT: 123", single["messages"][0].content)
    check("a ticker string carrying two symbols is treated as a batch",
          sorted(tool.seen) == ["AAPL", "NVDA"] and "=== AAPL ===" in packed["messages"][0].content, str(tool.seen))


def test_batch_reply_shares_one_context_budget_but_preserves_everything():
    """Eight full statements at the single-result cap would be ~5,000 tokens — bigger
    than the whole 3,500-token history window, so the window would drop them and the
    model would lose the data. The companies share one budget instead."""
    import nodes.chat_agent as ca
    from config import CHAT_TOOL_RESULT_MAX_CHARS
    big = {t: "x" * 2400 for t in _TICKS8}
    tool = _TimedTool(outcomes=big)
    saved = ca.TOOLS_BY_NAME
    ca.TOOLS_BY_NAME = {"financial_data_api": tool}
    try:
        out = ca.chat_tool_node(_batch_state({"tickers": _TICKS8}))
    finally:
        ca.TOOLS_BY_NAME = saved
    reply = out["messages"][0].content
    check("the merged reply stays within about twice the single-result cap",
          len(reply) < CHAT_TOOL_RESULT_MAX_CHARS * 2 + 900, f"{len(reply)} chars")
    check("every company is still present (none squeezed out)", all(f"=== {t} ===" in reply for t in _TICKS8))
    check("cut data is marked", "[truncated]" in reply)
    check("the FULL text of each company is preserved for rescue / evaluation",
          all(len(r["result"]) == 2400 for r in out["successful_tool_results"]))


def test_tool_wrapper_still_works_directly_and_rejects_an_empty_call():
    import tools as tools_pkg
    seen = []
    saved = tools_pkg.financial_data_api_fn
    tools_pkg.financial_data_api_fn = lambda **kw: (seen.append(kw["ticker"]), f"data {kw['ticker']}")[1]
    try:
        single = tools_pkg.financial_data_api.invoke({"ticker": "MSFT"})
        many = tools_pkg.financial_data_api.invoke({"tickers": ["AAPL", "NVDA"]})
        empty = tools_pkg.financial_data_api.invoke({})
    finally:
        tools_pkg.financial_data_api_fn = saved
    check("backward compatible: a single ticker behaves exactly as before", single == "data MSFT", single)
    check("called directly, a tickers list still fans out and merges", "=== AAPL ===" in many and "=== NVDA ===" in many, many)
    check("a call naming nothing gets a clear error, not a crash", "provide `ticker`" in empty, empty)


def test_extract_tickers_returns_every_company_in_order():
    from nodes.intent_router import extract_ticker_from_text, extract_tickers_from_text
    q = "Show me the market cap and P/E of Apple, Microsoft, Google, Amazon, Meta, Nvidia, Tesla and Netflix."
    check("all eight, in the order written", extract_tickers_from_text(q) == _TICKS8, str(extract_tickers_from_text(q)))
    check("the single-ticker function is unchanged (earliest only)", extract_ticker_from_text(q) == "AAPL")
    check("P/E is not mistaken for a ticker", "P" not in extract_tickers_from_text("what is the P/E of Apple"))
    check("nothing recognised -> an empty list, never a guess", extract_tickers_from_text("What is India's GDP growth?") == [])
    check("symbols and names mix, duplicates removed",
          extract_tickers_from_text("compare TSLA and Tesla and NVDA") == ["TSLA", "NVDA"],
          str(extract_tickers_from_text("compare TSLA and Tesla and NVDA")))


def test_validate_brief_normalises_and_trusts_its_own_name_table():
    from nodes.query_understanding import validate_brief
    raw = {"in_scope": True, "intent": "weird", "metrics": ["market cap"] * 20,
           "entities": [
               {"name": "Microsoft", "type": "company", "ticker": "AAPL"},          # the model's memory is wrong
               {"name": "Tata Motors", "type": "company", "ticker": "tatamotors.ns"},  # not in our table
               {"name": "Evil", "type": "company", "ticker": "DROP TABLE;"},         # not shaped like a ticker
               {"name": "India", "type": "country", "ticker": "IND"},                # countries never get one
               {"name": "", "type": "company"},                                       # unusable
           ]}
    b = validate_brief(raw, "q")
    ents = {e["name"]: e for e in b["entities"]}
    check("an unknown intent becomes chat", b["intent"] == "chat")
    check("our own table beats the model's ticker (Microsoft stays MSFT, verified)",
          ents["Microsoft"]["ticker"] == "MSFT" and ents["Microsoft"]["verified"] is True, str(ents["Microsoft"]))
    check("a plausible ticker for an unknown company is kept but marked UNVERIFIED",
          ents["Tata Motors"]["ticker"] == "TATAMOTORS.NS" and ents["Tata Motors"]["verified"] is False, str(ents["Tata Motors"]))
    check("a ticker that is not ticker-shaped is dropped", ents["Evil"]["ticker"] is None)
    check("a country never carries a ticker", ents["India"]["ticker"] is None)
    check("an entity with no name is discarded", len(b["entities"]) == 4, str(len(b["entities"])))
    check("metrics are capped", len(b["metrics"]) <= 8, str(len(b["metrics"])))
    check("garbage input yields None, never an exception",
          validate_brief("not a dict", "q") is None and validate_brief(None, "q") is None)


def test_the_model_may_propose_a_question_but_code_decides_if_it_is_asked():
    """
    The user's complaint: it should ask when something is missing. The design:
    the MODEL notices the gap, the CODE decides whether it is genuinely blocking —
    so a missing year becomes a stated assumption, while "compare them" with no
    companies and no context stops with a question.
    """
    from nodes.query_understanding import validate_brief
    blocked = validate_brief({"entities": [], "missing": ["which companies"], "ask": "Which companies?"}, "Compare their revenue growth", [])
    check("no subject anywhere -> the question IS asked", blocked["ask"] == "Which companies?", str(blocked))

    named = validate_brief({"entities": [{"name": "Tesla", "type": "company", "ticker": "TSLA"}],
                            "missing": ["which year"], "ask": "Which year?"}, "How is Tesla doing?", [])
    check("a company is named but no year -> NOT asked", named["ask"] is None, str(named))
    check("it becomes a stated assumption instead",
          any("year" in a and "latest" in a for a in named["assumptions"]), str(named["assumptions"]))

    ctx = validate_brief({"entities": [], "uses_previous_context": True, "missing": ["which company"], "ask": "Which company?"},
                         "and its net income?", ["MSFT"])
    check("a follow-up that the conversation can resolve is NOT asked", ctx["ask"] is None, str(ctx))

    local = validate_brief({"entities": [], "missing": ["company"], "ask": "Which company?"}, "revenue of NVDA?", [])
    check("a ticker the local rules can see overrides the model's 'missing'", local["ask"] is None, str(local))

    synth = validate_brief({"entities": [], "missing": ["which companies"]}, "compare them", [])
    check("missing items with no question written get one composed", synth["ask"] and "which companies" in synth["ask"], str(synth))

    oos = validate_brief({"in_scope": False, "entities": [], "missing": ["x"], "ask": "What?"}, "who won the match", [])
    check("an out-of-scope message is never asked a question", oos["ask"] is None and oos["in_scope"] is False)

    only_period = validate_brief({"entities": [], "missing": ["time period"], "ask": "Which period?"}, "tell me about inflation", [])
    check("a period-only gap is never a question (a default always exists)", only_period["ask"] is None, str(only_period))


def test_understand_query_parses_fenced_json_and_never_raises():
    from nodes.query_understanding import understand_query
    calls, restore = _scripted_dispatch(
        ['```json\n' + _brief_json(entities=[{"name": "Apple", "type": "company", "ticker": "AAPL"}], metrics=["price"]) + '\n```',
         "no json here at all", RuntimeError("provider down")], [])
    try:
        ok = understand_query("What is Apple's price?")
        garbage = understand_query("What is Apple's price?")
        boom = understand_query("What is Apple's price?")
    finally:
        restore()
    check("a fenced JSON reply is parsed", ok is not None and ok["entities"][0]["ticker"] == "AAPL", str(ok))
    check("the brief says where it came from", ok["source"] == "front_door")
    check("a reply with no JSON gives None (the router then uses its rules)", garbage is None)
    check("a provider error gives None, not an exception", boom is None)
    sent = str(calls["front"][0][1].content)
    check("the model is shown the user's message verbatim", "What is Apple's price?" in sent, sent)


def test_brief_for_prompt_steers_batching_and_flags_unverified_tickers():
    from nodes.query_understanding import brief_for_prompt, validate_brief
    multi = validate_brief({"entities": [{"name": n, "type": "company", "ticker": t} for n, t in zip(_NAMES8[:3], _TICKS8[:3])],
                            "metrics": ["market cap"]}, "q")
    text = brief_for_prompt(multi)
    check("two or more companies -> an explicit instruction to use ONE tickers call",
          "ONE financial_data_api call" in text and "tickers=['AAPL', 'MSFT', 'GOOGL']" in text, text)
    check("verified tickers carry no question mark", "Apple (AAPL)" in text and "(AAPL?)" not in text)
    odd = validate_brief({"entities": [{"name": "Tata Motors", "type": "company", "ticker": "TATAMOTORS.NS"}]}, "q")
    check("an unconfirmed ticker is flagged so the agent checks the returned company name",
          "(TATAMOTORS.NS?)" in brief_for_prompt(odd) and "unconfirmed" in brief_for_prompt(odd))
    country = validate_brief({"entities": [{"name": "India", "type": "country"}], "metrics": ["GDP growth"]}, "q")
    check("a country appears as a subject, not as a company", "India (country)" in brief_for_prompt(country))
    check("nothing useful to say -> an empty string (no tokens spent)", brief_for_prompt(None) == "" and brief_for_prompt({"entities": []}) == "")


def test_prompt_lists_all_companies_and_drops_the_single_company_anchor():
    """The old prompt said 'The company currently under discussion is AAPL' for an
    eight-company question, steering the model to that one company."""
    import nodes.chat_agent as ca
    multi = ca._system_prompt("AAPL", True, active_tickers=_TICKS8)
    check("all eight companies are named", ", ".join(_TICKS8) in multi)
    check("the single-company sentence is gone", "currently under discussion is **AAPL**" not in multi)
    check("ambiguous pronouns are called out", "ambiguous with several companies" in multi)
    single = ca._system_prompt("MSFT", True, active_tickers=["MSFT"])
    check("one company keeps the original wording", "currently under discussion is **MSFT**" in single)
    check("calls made the old way (positional, no new arguments) still work", "Today is" in ca._system_prompt(None, True))


def test_router_with_the_understanding_step_on_handles_eight_companies():
    import config as _c
    from nodes.intent_router import intent_router_node, route_after_intent
    calls, restore = _scripted_dispatch([_brief_json(
        entities=[{"name": n, "type": "company", "ticker": t} for n, t in zip(_NAMES8, _TICKS8)],
        metrics=["market cap", "P/E"])], [])
    _c.FRONT_DOOR_ENABLED = True
    try:
        out = intent_router_node({"messages": [HumanMessage(content="Show me the market cap and P/E of Apple, Microsoft, Google, Amazon, Meta, Nvidia, Tesla and Netflix.")]})
    finally:
        _c.FRONT_DOOR_ENABLED = False
        restore()
    check("routed via the understanding step", out["intent_source"] == "front_door", out["intent_source"])
    check("all eight companies are kept in state (it was the first one only)", out["active_tickers"] == _TICKS8, str(out["active_tickers"]))
    check("the primary ticker is still set for older readers", out["active_ticker"] == "AAPL")
    check("the brief is stored for the agent", out["brief"] and len(out["brief"]["entities"]) == 8)
    check("it goes on to the agent", route_after_intent(out) == "agent")


def test_router_asks_when_blocked_and_the_agent_never_runs():
    import config as _c
    from nodes.intent_router import intent_router_node, route_after_intent
    calls, restore = _scripted_dispatch([_brief_json(missing=["which companies"], ask="Which companies should I compare?")], [])
    _c.FRONT_DOOR_ENABLED = True
    try:
        out = intent_router_node({"messages": [HumanMessage(content="Compare their revenue growth")]})
    finally:
        _c.FRONT_DOOR_ENABLED = False
        restore()
    check("intent is CLARIFY", out["intent"] == "CLARIFY", out["intent"])
    check("the question to the user is the turn's reply", out["messages"][0].content == "Which companies should I compare?")
    check("the graph ends the turn there", route_after_intent(out) == "end")
    check("it leaves turn_status empty so the evaluator does not score a question", out["turn_status"] == "")
    check("a stale report offer cannot survive a clarification", out["pending_report_query"] is None)


def test_router_redirects_out_of_scope_and_keeps_the_company_in_focus():
    import config as _c
    from nodes.intent_router import intent_router_node, route_after_intent
    calls, restore = _scripted_dispatch([_brief_json(in_scope=False)], [])
    _c.FRONT_DOOR_ENABLED = True
    try:
        out = intent_router_node({"messages": [HumanMessage(content="Who won the cricket match?")],
                                  "active_tickers": ["MSFT"], "active_ticker": "MSFT"})
    finally:
        _c.FRONT_DOOR_ENABLED = False
        restore()
    check("intent is OUT_OF_SCOPE", out["intent"] == "OUT_OF_SCOPE", out["intent"])
    check("it takes the cheap no-tools path", route_after_intent(out) == "smalltalk")
    check("an off-topic aside does not wipe the company being discussed", out["active_tickers"] == ["MSFT"], str(out["active_tickers"]))


def test_router_follow_ups_keep_context_but_new_topics_do_not():
    """A question about India's GDP used to arrive tagged with Apple, because a
    message with no ticker inherited the previous one."""
    import config as _c
    from nodes.intent_router import intent_router_node
    calls, restore = _scripted_dispatch([
        _brief_json(uses_previous_context=True, metrics=["net income"]),
        _brief_json(entities=[{"name": "India", "type": "country"}], metrics=["GDP growth"]),
        _brief_json(metrics=["inflation"]),
    ], [])
    prior = {"active_tickers": ["MSFT"], "active_ticker": "MSFT"}
    _c.FRONT_DOOR_ENABLED = True
    try:
        follow = intent_router_node({"messages": [HumanMessage(content="and its net income?")], **prior})
        topic = intent_router_node({"messages": [HumanMessage(content="What is India's GDP growth?")], **prior})
        bare = intent_router_node({"messages": [HumanMessage(content="What is inflation doing?")], **prior})
    finally:
        _c.FRONT_DOOR_ENABLED = False
        restore()
    check("a genuine follow-up keeps the company", follow["active_tickers"] == ["MSFT"], str(follow["active_tickers"]))
    check("a question about a country does not inherit the company", topic["active_tickers"] == [], str(topic["active_tickers"]))
    check("neither does an unrelated macro question", bare["active_tickers"] == [], str(bare["active_tickers"]))
    check("the country is a subject in the brief", topic["brief"]["entities"][0]["type"] == "country")


def test_router_skips_the_understanding_call_when_rules_already_settled_the_turn():
    import config as _c
    from nodes.intent_router import intent_router_node
    calls, restore = _scripted_dispatch([_brief_json()] * 5, [])
    _c.FRONT_DOOR_ENABLED = True
    try:
        greet = intent_router_node({"messages": [HumanMessage(content="hi")]})
        yes = intent_router_node({"messages": [HumanMessage(content="yes")], "pending_report_query": "report on Apple"})
        no = intent_router_node({"messages": [HumanMessage(content="no")], "pending_report_query": "report on Apple"})
        rep = intent_router_node({"messages": [HumanMessage(content="generate a full research report on Apple")]})
    finally:
        _c.FRONT_DOOR_ENABLED = False
        restore()
    check("no understanding call for a greeting, a yes, a no, or an explicit report request",
          len(calls["front"]) == 0, f"{len(calls['front'])} calls")
    check("those turns still route correctly",
          [greet["intent"], yes["intent"], no["intent"], rep["intent"]] == ["SMALLTALK", "CONFIRM_REPORT", "CHAT", "REPORT"],
          str([greet["intent"], yes["intent"], no["intent"], rep["intent"]]))


def test_router_falls_back_to_rules_when_the_step_is_off_or_fails():
    import config as _c
    from nodes.intent_router import intent_router_node
    calls, restore = _scripted_dispatch(["not json", RuntimeError("down")], [])
    try:
        _c.FRONT_DOOR_ENABLED = False
        off = intent_router_node({"messages": [HumanMessage(content="What is Microsoft's revenue?")]})
        _c.FRONT_DOOR_ENABLED = True
        bad = intent_router_node({"messages": [HumanMessage(content="What is Microsoft's revenue?")]})
        err = intent_router_node({"messages": [HumanMessage(content="What is Microsoft's revenue?")]})
    finally:
        _c.FRONT_DOOR_ENABLED = False
        restore()
    check("switched off: no model call, rules route it", len(calls["front"]) == 2 and off["intent_source"] == "rules", f"{off['intent_source']} {len(calls['front'])}")
    check("garbage from the model: rules take over and the turn still works",
          bad["intent"] == "CHAT" and bad["intent_source"] == "rules" and bad["active_tickers"] == ["MSFT"], str(bad["intent_source"]))
    check("a provider error: same fallback", err["intent"] == "CHAT" and err["active_tickers"] == ["MSFT"])
    check("no brief is invented when the step failed", bad["brief"] is None and err["brief"] is None)


def test_end_to_end_eight_companies_take_one_round_and_the_rest_of_the_paths_work():
    """Through the real compiled graph: your 8-company question, a blocked question,
    an off-topic one, and the 'which year?' case."""
    import config as _c
    front = [
        _brief_json(entities=[{"name": n, "type": "company", "ticker": t} for n, t in zip(_NAMES8, _TICKS8)], metrics=["market cap", "P/E"]),
        _brief_json(missing=["which companies"], ask="Which companies, and over what period?"),
        _brief_json(in_scope=False),
        _brief_json(entities=[{"name": "Tesla", "type": "company", "ticker": "TSLA"}], missing=["which year"], ask="Which year?"),
    ]
    agent = [AIMessage(content="", tool_calls=[{"name": "financial_data_api", "args": {"tickers": _TICKS8, "fields": ["market_cap", "pe"]}, "id": "c1"}]),
             AIMessage(content="Here are all eight."),
             AIMessage(content="Tesla, latest data.")]
    fakes = install_fake_tools({"financial_data_api": "Market Cap: 1 | P/E: 2"})
    graph = fresh_graph()
    calls, restore = _scripted_dispatch(front, agent)
    _c.FRONT_DOOR_ENABLED = True
    try:
        a = run(graph, "Show me the market cap and P/E of Apple, Microsoft, Google, Amazon, Meta, Nvidia, Tesla and Netflix.", thread="fd-a")
        agent_calls_after_a = len(calls["agent"])
        b = run(graph, "Compare their revenue growth", thread="fd-b")
        agent_calls_after_b = len(calls["agent"])
        c = run(graph, "Who won the cricket match?", thread="fd-c")
        d = run(graph, "How is Tesla doing?", thread="fd-d")
    finally:
        _c.FRONT_DOOR_ENABLED = False
        restore()
    tool_msgs = [m for m in a["messages"] if isinstance(m, ToolMessage)]
    check("eight companies: ONE tool round", a.get("tool_loops") == 1, str(a.get("tool_loops")))
    check("eight companies: all eight fetched", fakes["financial_data_api"].call_count == 8, str(fakes["financial_data_api"].call_count))
    check("the model received one merged tool reply", len(tool_msgs) == 1 and "=== NVDA ===" in tool_msgs[0].content)
    check("the agent was told the topic is the eight companies, not Apple",
          "currently under discussion is **AAPL**" not in str(calls["agent"][0][0].content)
          and ", ".join(_TICKS8) in str(calls["agent"][0][0].content))
    check("the user's wording is stored exactly as typed",
          str(a["messages"][0].content).startswith("Show me the market cap and P/E of Apple"))
    check("blocked question: the agent was not run", agent_calls_after_b == agent_calls_after_a, f"{agent_calls_after_a} -> {agent_calls_after_b}")
    check("blocked question: the reply is the question", str(b["messages"][-1].content) == "Which companies, and over what period?")
    check("off-topic: redirected with no tools", "financial research" in str(c["messages"][-1].content) and c.get("intent") == "OUT_OF_SCOPE")
    check("'which year?' is answered with an assumption, not a question",
          d.get("intent") == "CHAT" and "Which year?" not in str(d["messages"][-1].content))
    check("and the assumption reached the agent", "was not specified, so use the latest available" in str(calls["agent"][-1][0].content))


def test_out_of_scope_reply_falls_back_to_a_fixed_message_if_the_model_fails():
    import nodes.chat_agent as ca
    class Boom:
        def invoke(self, m): raise RuntimeError("down")
    saved = (ca.get_llm, ca.invoke_with_retry)
    ca.get_llm = lambda *a, **k: Boom()
    ca.invoke_with_retry = lambda llm, m, **k: llm.invoke(m)
    try:
        out = ca.smalltalk_node({"messages": [HumanMessage(content="who won the match")], "intent": "OUT_OF_SCOPE"})
        plain = ca.smalltalk_node({"messages": [HumanMessage(content="hi")], "intent": "SMALLTALK"})
    finally:
        ca.get_llm, ca.invoke_with_retry = saved
    check("the fixed redirect is used and mentions finance", "financial research assistant" in out["messages"][0].content, out["messages"][0].content)
    check("ordinary small talk keeps its own fallback", "What company are you looking at?" in plain["messages"][0].content)


def test_cache_keys_ignore_case_and_padding_but_not_meaning():
    """'msft', 'MSFT' and ' Msft ' were three cache entries for one request."""
    from database.cache_service import canonical_args, make_cache_key
    a = make_cache_key("financial_data_api", {"ticker": "msft", "statement_type": "Income", "period": "ANNUAL", "years": 2})
    b = make_cache_key("financial_data_api", {"ticker": " MSFT ", "statement_type": "income", "period": "annual", "years": 2})
    check("spelling variants of the same request share ONE key", a == b)
    check("a different company is a different key",
          a != make_cache_key("financial_data_api", {"ticker": "AAPL", "statement_type": "income", "period": "annual", "years": 2}))
    check("a different number of years is a different key",
          a != make_cache_key("financial_data_api", {"ticker": "MSFT", "statement_type": "income", "period": "annual", "years": 3}))
    check("argument order still does not matter",
          a == make_cache_key("financial_data_api", {"years": 2, "period": "annual", "statement_type": "income", "ticker": "MSFT"}))
    canon = canonical_args({"ticker": " tcs.ns", "filing_type": "10-K", "years": 3, "fields": ["a"]})
    check("only ticker / statement_type / period are touched; a filing type and non-strings are left alone",
          canon == {"ticker": "TCS.NS", "filing_type": "10-K", "years": 3, "fields": ["a"]}, str(canon))


def test_spelling_variants_make_one_live_call_and_one_company_row():
    import tools as tools_pkg
    from database.connection import init_db, query
    init_db(force=True)
    calls = {"n": 0}
    def fake(**kw):
        calls["n"] += 1
        return ("[FinData] Income Statement (Annual) \u2013 CANX\n\n  Period: 2025-06-30\n"
                "  Total Revenue      : 300\n")
    real = tools_pkg._findata
    tools_pkg._findata = fake
    try:
        for spelling in ("canx", " CANX ", "Canx", "CANX"):
            tools_pkg.financial_data_api_fn(spelling, "Income", "Annual", 1)
    finally:
        tools_pkg._findata = real
    rows = query("SELECT ticker FROM companies WHERE UPPER(ticker) = 'CANX'")
    check("four spellings, ONE live call", calls["n"] == 1, str(calls["n"]))
    check("and exactly one company row, stored upper-case (a lower-case spelling used to add a second)",
          [r["ticker"] for r in rows] == ["CANX"], str([dict(r) for r in rows]))


def test_a_fred_shortcut_and_the_real_series_code_share_one_cache_entry():
    import tools as tools_pkg
    from database.connection import init_db
    init_db(force=True)
    calls = {"n": 0}
    def fake(**kw):
        calls["n"] += 1
        return ("[FRED] Series: GDP  (raw series ID)\n" + "=" * 60 + "\nDate  Value\n" + "-" * 30 + "\n2024-01-01   29,380.99\n")
    real = tools_pkg._fred
    tools_pkg._fred = fake
    try:
        tools_pkg.fred_series_fn("gdp")
        tools_pkg.fred_series_fn("GDP")
        tools_pkg.fred_series_fn(" gdp ")
    finally:
        tools_pkg._fred = real
    check("'gdp', 'GDP' and ' gdp ' made ONE live call", calls["n"] == 1, str(calls["n"]))


def test_cache_writes_purge_expired_rows_at_most_once_an_hour():
    """purge_expired existed but nothing ever called it, so response_cache only grew."""
    import time as _time
    import database.cache_service as cs
    from database.connection import execute, init_db, query
    init_db(force=True)
    old_flag = cs._last_purge

    def put(key, expires):
        execute("INSERT INTO response_cache (cache_key, tool_name, payload, created_at, expires_at) VALUES (?,?,?,?,?)",
                (key, "t", "p", "2000-01-01T00:00:00+00:00", expires))
    keys = lambda: {r["cache_key"] for r in query("SELECT cache_key FROM response_cache")}
    try:
        put("expired-1", "2000-01-02T00:00:00+00:00")
        put("fresh-1", "2999-01-01T00:00:00+00:00")
        cs._last_purge = 0.0
        cs.set_cached("new-1", "t", "payload", "news")
        check("a write triggered the purge: the expired row is gone", "expired-1" not in keys(), str(keys()))
        check("fresh rows and the new row are untouched", {"fresh-1", "new-1"} <= keys(), str(keys()))

        put("expired-2", "2000-01-02T00:00:00+00:00")
        cs.set_cached("new-2", "t", "payload", "news")
        check("a second write within the hour does NOT purge again", "expired-2" in keys(), str(keys()))

        cs._last_purge = _time.time() - cs._PURGE_INTERVAL_S - 1
        cs.set_cached("new-3", "t", "payload", "news")
        check("after the interval it purges again", "expired-2" not in keys(), str(keys()))

        saved = cs.purge_expired
        cs.purge_expired = lambda: (_ for _ in ()).throw(RuntimeError("db locked"))
        cs._last_purge = 0.0
        try:
            cs.set_cached("new-4", "t", "payload", "news")
            raised = False
        except Exception:
            raised = True
        finally:
            cs.purge_expired = saved
        check("a failing purge can never break a cache write", raised is False and "new-4" in keys(), str(keys()))
    finally:
        cs._last_purge = old_flag


def test_the_code_base_has_no_syntax_newer_than_python_3_9():
    """
    A real failure: `list[str] | None` in an annotation works on Python 3.10+ but
    raises TypeError when the module is imported on 3.9 — the server could not even
    start on a 3.9 install, while the test suite (run on a newer Python) stayed
    green. This reads every source file and fails on the constructs that break
    3.9, so the next one is caught here instead of at startup.
    """
    import ast
    root = Path(__file__).resolve().parent.parent
    problems = []

    def has_union(node):
        return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) for n in ast.walk(node))

    files = [p for p in root.rglob("*.py")
             if not any(part in ("__pycache__", "output", "sec_data", ".git") for part in p.parts)]
    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, feature_version=(3, 9))
        except SyntaxError as exc:
            problems.append(f"{path.name}:{exc.lineno} syntax newer than 3.9 ({exc.msg})")
            continue
        postponed = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                        and any(a.name == "annotations" for a in n.names) for n in tree.body)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not postponed:
                anns = [a.annotation for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs if a.annotation]
                anns += [x for x in (node.returns, getattr(node.args.vararg, "annotation", None),
                                     getattr(node.args.kwarg, "annotation", None)) if x]
                if any(has_union(a) for a in anns):
                    problems.append(f"{path.name}:{node.lineno} 'X | Y' annotation needs 3.10+ (no __future__ annotations)")
            elif isinstance(node, ast.AnnAssign) and not postponed and has_union(node.annotation):
                problems.append(f"{path.name}:{node.lineno} 'X | Y' variable annotation needs 3.10+")
            elif (isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("isinstance", "issubclass")
                  and any(has_union(a) for a in node.args[1:])):
                problems.append(f"{path.name}:{node.lineno} isinstance with 'X | Y' needs 3.10+")
            elif (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "zip"
                  and any(k.arg == "strict" for k in node.keywords)):
                problems.append(f"{path.name}:{node.lineno} zip(strict=) needs 3.10+")
    check("every file parses as Python 3.9 and avoids 3.10+-only annotations", problems == [], "; ".join(problems[:5]))
    check("the scan really looked at the code base", len(files) > 40, str(len(files)))


def test_sec_long_term_debt_does_not_masquerade_as_total_debt():
    """
    SEC's LongTermDebt concept used to be stored in balance_sheets.total_debt —
    the column yfinance's 'Total Debt' (which also counts short-term debt and
    leases) lands in. total_debt is one of the fields the contradiction
    detector compares across sources, so two DIFFERENT measures were compared
    as one and could raise a false conflict, with SEC (tier A) 'winning'.
    """
    from database.connection import init_db, query, execute
    from database.normalizer import (normalize_sec_xbrl_text, normalize_yfinance_text,
                                     record_source)
    from database.schema import DATA_MIGRATIONS
    from utils.contradiction_detector import find_contradictions_from_state

    init_db(force=True)
    sec = ("[SEC XBRL] LTDTEST 10-K\n"
           "  Long-Term Debt                                :       $96.70B  (FY end: 2024-09-28)\n"
           "  Shareholders Equity                           :       $56.95B  (FY end: 2024-09-28)\n")
    sec_src = record_source(provider="sec", source_type="filing")
    normalize_sec_xbrl_text("LTDTEST", sec, sec_src, period_type="annual")
    row = query("SELECT long_term_debt, total_debt FROM balance_sheets WHERE company_id IN "
                "(SELECT id FROM companies WHERE ticker='LTDTEST')")[0]
    check("SEC value lands in long_term_debt", abs(row["long_term_debt"] - 96.7e9) < 1, str(dict(row)))
    check("total_debt is left alone (NULL) by the SEC path", row["total_debt"] is None, str(dict(row)))

    yf = ("[FinData] Balance Sheet (Annual) \u2013 LTDTEST\n\n  Period: 2024-09-28\n"
          "  Total Debt          : 106600000000\n  Shareholder Equity  : 56950000000\n")
    yf_src = record_source(provider="yfinance", source_type="api")
    normalize_yfinance_text("LTDTEST", yf, yf_src)
    state = {"successful_tool_results": [{"tool": "financial_data_api",
             "args": {"ticker": "LTDTEST"}, "result": yf, "status": "success"}],
             "active_ticker": "LTDTEST"}
    flagged = [c for c in find_contradictions_from_state(state) if c.get("field") == "total_debt"]
    check("no false total_debt contradiction between SEC long-term and yfinance total",
          flagged == [], str(flagged))

    cid = query("SELECT id FROM companies WHERE ticker='LTDTEST'")[0]["id"]
    execute("INSERT INTO balance_sheets (company_id,period_end,period_type,total_debt,source_id,updated_at) "
            "VALUES (?,?,?,?,?,?)", (cid, "2022-09-30", "annual", 90e9, sec_src, "x"))
    for _ in range(2):  # idempotent: second pass must change nothing
        for stmt in DATA_MIGRATIONS:
            execute(stmt)
    legacy = query("SELECT total_debt, long_term_debt FROM balance_sheets WHERE company_id=? "
                   "AND period_end='2022-09-30'", (cid,))[0]
    check("legacy SEC row (value in total_debt) is relocated to long_term_debt",
          legacy["total_debt"] is None and legacy["long_term_debt"] == 90e9, str(dict(legacy)))


def test_fred_normalizer_handles_comma_formatted_values():
    """
    Regression test for a real bug found via manual DB inspection: the
    macro_series table had rows (series registered correctly) but
    macro_observations was completely EMPTY (the actual GDP numbers never
    landed). Root cause: the parsing regex's value pattern [\\-\\d.]+ has no
    comma in its character class, but tools/fred.py formats every value
    with thousand-separators ("29,929.55", "7,037,663.00"). Every
    GDP-scale observation silently failed to match and was dropped with no
    error — the exact kind of silent-swallow bug that's hardest to notice,
    since nothing crashes and nothing logs; the table just stays empty.
    """
    from database.connection import init_db, query
    from database.normalizer import normalize_fred_series, record_source

    init_db(force=True)
    src = record_source(provider="fred", source_type="api")

    # Real FRED output shape, including a double-comma value (NA000334Q-style)
    sample = (
        "[FRED] Series: GDPA  (raw series ID)\n"
        "============================================================\n"
        "Date                    Value\n"
        "------------------------------\n"
        "2024-01-01          29,380.99\n"
        "2023-01-01          27,825.89\n"
    )
    written = normalize_fred_series("GDPATEST", "GDPATEST", sample, src)
    check("comma-formatted FRED values are written, not silently dropped",
          written == 2, f"wrote {written} rows, expected 2")

    rows = query(
        "SELECT date, value FROM macro_observations WHERE series_id IN "
        "(SELECT id FROM macro_series WHERE series_code='GDPATEST') ORDER BY date"
    )
    check("value parsed correctly with commas stripped (29,380.99 -> 29380.99)",
          len(rows) == 2 and abs(rows[1]["value"] - 29380.99) < 0.01,
          str([dict(r) for r in rows]))

    # Double-comma case: 7,037,663.00 (NA000334Q quarterly GDP scale)
    sample_big = (
        "Date                    Value\n"
        "------------------------------\n"
        "2024-01-01          7,037,663.00\n"
    )
    written_big = normalize_fred_series("BIGTEST", "BIGTEST", sample_big, src)
    rows_big = query(
        "SELECT value FROM macro_observations WHERE series_id IN "
        "(SELECT id FROM macro_series WHERE series_code='BIGTEST')"
    )
    check("double-comma values (millions scale) parsed correctly",
          written_big == 1 and abs(rows_big[0]["value"] - 7037663.00) < 0.01,
          str([dict(r) for r in rows_big]))


def test_partial_response_does_not_claim_full_refetch():
    """
    Regression test for a real user-facing accuracy issue: the partial
    result message used to say 're-running costs another set of API
    calls', implying the whole turn would refetch data. That's wrong — the
    tool calls shown already succeeded and are cached, so re-asking reuses
    them with zero new calls to the data providers. Only the SYNTHESIS step
    (the LLM call that failed) genuinely needs to run again. The message
    must accurately distinguish these, not blanket-claim everything refetches.
    """
    from utils.result_formatter import build_partial_response

    successful = [{
        "tool": "financial_data_api", "args": {"ticker": "MSFT"},
        "result": '{"Market Cap": 3894773022720}', "status": "success",
    }]

    message = build_partial_response(successful, Exception("Connection error."))

    check("does NOT claim re-running costs another set of API calls",
          "costs another set of" not in message, message)
    check("does NOT claim nothing will be reused",
          "reuse nothing" not in message, message)
    check("DOES say the data will be reused from cache",
          "reuse the data above from cache" in message, message)
    check("DOES say no new API calls to data providers",
          "no new API" in message, message)
    check("DOES correctly say synthesis/analysis needs a fresh attempt",
          "fresh attempt" in message and "analysis" in message, message)


def test_calculator_financial_metrics_still_work():
    """Verify the arithmetic additions did not break existing financial metrics."""
    from tools.calculator import calculator

    check("cagr still works",
          "10." in calculator(metric="cagr", start_value=100, end_value=196, years=7),
          calculator(metric="cagr", start_value=100, end_value=196, years=7))
    check("gross_margin with gross_profit still works",
          "40.00%" in calculator(metric="gross_margin", revenue=1000, gross_profit=400),
          calculator(metric="gross_margin", revenue=1000, gross_profit=400))
    check("roe still works",
          "20.00%" in calculator(metric="roe", net_income=100, shareholder_equity=500),
          calculator(metric="roe", net_income=100, shareholder_equity=500))
    check("gross_margin has no dead code bug (gross_profit path returns correctly)",
          "Gross Margin = 25.00%" in calculator(metric="gross_margin", revenue=800, gross_profit=200),
          calculator(metric="gross_margin", revenue=800, gross_profit=200))


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
        test_transient_connection_error_retries_instead_of_failing_immediately,
        test_transient_connection_error_eventually_gives_up,
        test_rescue_synthesis_recovers_when_retry_replays_bad_history,
        test_rescue_synthesis_does_not_fire_for_unrelated_errors,
        test_sec_xbrl_normalizer_parses_scaled_values,
        test_sec_normalizer_skips_na_lines_safely,
        test_calculator_arithmetic_operations,
        test_calculator_arithmetic_error_handling,
        test_overview_block_populates_companies_and_valuation_snapshots,
        test_cashflow_dividends_and_buybacks_are_stored,
        test_schema_migration_upgrades_an_existing_database,
        test_fred_metadata_is_stored_and_shortcuts_resolve,
        test_dividend_yield_is_correct_under_either_yfinance_convention,
        test_rescue_prompt_carries_context_contradictions_and_history,
        test_rescue_data_budget_is_shared_fairly_and_truncation_is_marked,
        test_rescue_history_is_plain_text_without_tool_traffic,
        test_rescued_turn_is_tagged_and_normal_turn_is_not,
        test_evaluator_records_synthesis_mode_in_metadata,
        test_statement_currency_is_tagged_and_stored,
        test_trading_and_reporting_currency_are_kept_separate,
        test_refresh_fills_currency_on_rows_that_already_exist,
        test_sec_rows_are_tagged_usd,
        test_currency_lookup_separates_trading_from_reporting_and_does_not_cache_failures,
        test_fx_rate_tool_direct_inverse_errors_and_caching,
        test_fx_tool_is_chat_only_and_prompt_teaches_currency_and_suffix_rules,
        test_exact_text_cache_is_checked_before_the_db_rebuild,
        test_db_rebuild_keeps_currency_tag_and_cashflow_fields,
        test_all_is_never_served_as_a_partial_db_rebuild,
        test_refetch_replaces_rows_and_never_duplicates_a_fiscal_year,
        test_legacy_duplicates_are_ignored_by_reads_and_removed_by_migration,
        test_classifier_trusts_failure_keywords_only_on_error_shaped_text,
        test_chat_transient_failure_is_retried_in_code_after_a_pause,
        test_chat_systemic_failure_withdraws_the_tool_for_the_turn,
        test_chat_bad_args_are_not_blindly_retried_and_repeats_are_refused,
        test_chat_empty_result_tells_the_model_to_stop,
        test_chat_raised_exceptions_are_classified_from_their_text,
        test_withdrawn_tools_reset_on_the_next_user_message,
        test_report_fetch_transient_retry_waits_first,
        test_report_fetch_bad_args_get_one_model_corrected_retry,
        test_report_fetch_systemic_and_empty_are_never_retried,
        test_correct_args_helper_only_spends_a_model_call_when_it_can_help,
        test_identical_concurrent_requests_share_one_live_call,
        test_different_requests_are_not_serialized_by_coalescing,
        test_a_failing_leader_never_strands_or_poisons_the_waiters,
        test_an_error_result_is_shared_not_retried_by_every_waiter,
        test_a_waiter_gives_up_on_a_stuck_leader,
        test_rate_limit_wait_is_not_spent_after_the_last_attempt,
        test_daily_limit_falls_back_to_another_model_but_per_minute_limit_waits,
        test_report_section_prompt_contains_the_users_original_request,
        test_fair_share_and_the_judge_sees_every_tool_result_evenly,
        test_normalize_tickers_handles_lists_strings_names_and_the_cap,
        test_one_batch_call_is_one_round_with_one_reply_and_per_company_records,
        test_one_failing_company_does_not_fail_the_batch_and_is_named,
        test_batch_cap_and_unrecognised_names_are_reported,
        test_single_ticker_calls_are_untouched_and_a_packed_string_is_a_batch,
        test_batch_reply_shares_one_context_budget_but_preserves_everything,
        test_tool_wrapper_still_works_directly_and_rejects_an_empty_call,
        test_extract_tickers_returns_every_company_in_order,
        test_validate_brief_normalises_and_trusts_its_own_name_table,
        test_the_model_may_propose_a_question_but_code_decides_if_it_is_asked,
        test_understand_query_parses_fenced_json_and_never_raises,
        test_brief_for_prompt_steers_batching_and_flags_unverified_tickers,
        test_prompt_lists_all_companies_and_drops_the_single_company_anchor,
        test_router_with_the_understanding_step_on_handles_eight_companies,
        test_router_asks_when_blocked_and_the_agent_never_runs,
        test_router_redirects_out_of_scope_and_keeps_the_company_in_focus,
        test_router_follow_ups_keep_context_but_new_topics_do_not,
        test_router_skips_the_understanding_call_when_rules_already_settled_the_turn,
        test_router_falls_back_to_rules_when_the_step_is_off_or_fails,
        test_end_to_end_eight_companies_take_one_round_and_the_rest_of_the_paths_work,
        test_out_of_scope_reply_falls_back_to_a_fixed_message_if_the_model_fails,
        test_cache_keys_ignore_case_and_padding_but_not_meaning,
        test_spelling_variants_make_one_live_call_and_one_company_row,
        test_a_fred_shortcut_and_the_real_series_code_share_one_cache_entry,
        test_cache_writes_purge_expired_rows_at_most_once_an_hour,
        test_the_code_base_has_no_syntax_newer_than_python_3_9,
        test_sec_long_term_debt_does_not_masquerade_as_total_debt,
        test_fred_normalizer_handles_comma_formatted_values,
        test_partial_response_does_not_claim_full_refetch,
        test_calculator_financial_metrics_still_work,
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
