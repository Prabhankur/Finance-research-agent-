"""
demo_parallel.py — see width (parallel) vs depth (sequential) vs the BATCH fix.

Run from the project root:   python demo_parallel.py

Uses the REAL chat_agent_node / chat_tool_node with a scripted "model" and a fake
slow tool (0.5 s per call), so it needs no API key. It shows what the CODE does
for each way a model might ask for eight companies.
"""
import time
from langchain_core.messages import AIMessage, HumanMessage

import nodes.chat_agent as ca
from database.connection import init_db

init_db(force=False)
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "ORCL"]


class SlowTool:
    """Stands in for yfinance: every call takes 0.5 seconds."""
    def __init__(self):
        self.runs = 0
    def invoke(self, args):
        self.runs += 1
        time.sleep(0.5)
        return f"Market Cap for {args['ticker']}: 123"


def call(i, ticker):
    return {"name": "financial_data_api", "args": {"ticker": ticker}, "id": f"c{i}"}


def scenario(title, replies):
    tool = SlowTool()
    ca.TOOLS_BY_NAME = {"financial_data_api": tool}
    replies = list(replies)

    class FakeLLM:
        def bind_tools(self, *a, **k): return self
        def invoke(self, messages): return replies.pop(0)
    ca.get_llm = lambda *a, **k: FakeLLM()
    ca.invoke_with_retry = lambda llm, msgs, **k: llm.invoke(msgs)

    state = {"messages": [HumanMessage(content="compare these companies")], "user_input": "compare",
             "tool_loops": 0, "successful_tool_results": [], "tool_trace": [],
             "active_ticker": None, "disabled_tools": []}
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    started = time.time()
    while True:
        out = ca.chat_agent_node(state)
        state["messages"] += out.get("messages", [])
        if not getattr(state["messages"][-1], "tool_calls", None):
            break
        out2 = ca.chat_tool_node(state)
        state["messages"] += out2["messages"]
        state["tool_loops"] = out2["tool_loops"]
        state["tool_trace"] += out2["tool_trace"]
    print(f"--> companies fetched: {tool.runs} | rounds used: {state['tool_loops']} | "
          f"wall-clock: {time.time() - started:.1f}s")


final = AIMessage(content="Here is the comparison.")

scenario("A) one message, 8 SEPARATE calls  (width: the model must choose to do this)",
         [AIMessage(content="", tool_calls=[call(i, t) for i, t in enumerate(TICKERS[:8])]), final])

scenario("B) one call per message  (depth: what your real run did — 4 of 8 companies made it)",
         [AIMessage(content="", tool_calls=[call(i, TICKERS[i])]) for i in range(4)] + [final])

scenario("C) ONE call with tickers=[8]  (the fix: the tool's shape, not the model's habits)",
         [AIMessage(content="", tool_calls=[{"name": "financial_data_api",
                                              "args": {"tickers": TICKERS[:8]}, "id": "batch1"}]), final])

scenario("D) ONE call naming 10 companies  (cap: 8 fetched, the other 2 are reported)",
         [AIMessage(content="", tool_calls=[{"name": "financial_data_api",
                                              "args": {"tickers": TICKERS}, "id": "batch2"}]), final])
