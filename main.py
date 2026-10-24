"""
main.py  –  Entry point for the Autonomous Financial Report Agent.

Usage:
    python main.py --query "Analyse Apple AAPL for FY2024"
    python main.py --query "Analyse Microsoft MSFT for FY2024" --stream
    python main.py   # runs the default demo query

    # Launch Streamlit UI instead:
    streamlit run ui/app.py
"""

from __future__ import annotations
import argparse
import sys
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith tracing (must be set BEFORE importing langchain/langgraph) ──────
def _setup_langsmith() -> bool:
    api_key  = os.getenv("LANGCHAIN_API_KEY", "")
    tracing  = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    project  = os.getenv("LANGCHAIN_PROJECT", "financial-agent")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    if api_key and tracing == "true":
        os.environ["LANGCHAIN_TRACING_V2"]  = "true"
        os.environ["LANGCHAIN_API_KEY"]      = api_key
        os.environ["LANGCHAIN_PROJECT"]      = project
        os.environ["LANGCHAIN_ENDPOINT"]     = endpoint
        print(f"[LangSmith] ✓ Tracing enabled → project: '{project}'")
        return True
    else:
        print("[LangSmith] ✗ Tracing disabled.")
        return False


# ── Validate required environment variables ───────────────────────────────────
REQUIRED_KEYS = {
    "GROQ_API_KEY":      "Groq LLM (required – llama-3.3-70b-versatile)",
    "ALPHA_VANTAGE_KEY": "Alpha Vantage transcripts & financials",
    "TAVILY_API_KEY":    "Tavily web search",
    "FRED_API_KEY":      "FRED macro data (free at fred.stlouisfed.org/docs/api/api_key.html)",
}


def check_env() -> list[str]:
    """
    Warns about missing keys instead of exiting. A missing key no longer
    stops the run before it even starts — the fetch executor / replanner
    are specifically built to route around a data source that isn't
    available, so hard-exiting here would defeat that. GROQ_API_KEY is the
    one genuine hard requirement (nothing runs without an LLM at all), so
    that one still exits; the rest just print a warning and continue.

    Returns the list of missing optional keys, so the caller can pre-seed
    disabled_tools with the tools that depend on them.
    """
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]

    if "GROQ_API_KEY" in missing:
        print("\n✗ GROQ_API_KEY is missing — the agent cannot run at all without an LLM.")
        print("  Copy .env.example to .env and fill it in.\n")
        sys.exit(1)

    optional_missing = [k for k in missing if k != "GROQ_API_KEY"]
    if optional_missing:
        print("\n⚠  Missing optional environment variables — continuing anyway:")
        for k in optional_missing:
            print(f"   {k}  →  {REQUIRED_KEYS[k]}")
        print("  Any tool that depends on these will be marked unavailable for this")
        print("  run and the agent will route around it instead of using it.\n")

    return optional_missing


# Maps an env var to the tool(s) that can't function without it, so a
# missing key can pre-seed disabled_tools before the graph even starts.
_KEY_TO_TOOLS = {
    "ALPHA_VANTAGE_KEY": ["earnings_transcript"],
    "TAVILY_API_KEY":    ["web_search"],
    "FRED_API_KEY":      ["fred_series"],
}


# ── LangGraph config ──────────────────────────────────────────────────────────
#
# recursion_limit = max number of node executions before LangGraph force-stops.
# The fetch/compute executors each run their whole step list in one node
# call now (no more one-LLM-call-per-tool-step loop), so the only thing that
# can add extra node executions is a replan cycle — bounded by
# config.GLOBAL_REPLAN_BUDGET (each cycle = replanner + one re-entry into an
# executor, so 2 extra node executions per replan). Worst case is roughly
# 8 core nodes + (GLOBAL_REPLAN_BUDGET * 2). 100 is a generous safety margin.
#
from config import GRAPH_RECURSION_LIMIT
GRAPH_CONFIG = {"recursion_limit": GRAPH_RECURSION_LIMIT}


# ── Core agent runner ─────────────────────────────────────────────────────────

def run_agent(query: str, stream: bool = False,
              preseed_disabled_tools: Optional[list] = None,
              report_type: str = "") -> dict:
    """
    Runs the full financial agent pipeline for a given query.

    Args:
        query:  The user's financial research question.
        stream: If True, prints each node's output as it completes.
        preseed_disabled_tools: Tools known to be unavailable before the
            run even starts (e.g. a missing API key) — skips straight to
            "disabled" for these instead of discovering it via a live failure.
        report_type: Optional user-selected report archetype (one of
            config.REPORT_TYPES). If provided, skips the LLM classification
            step. Leave as "" (default) to let the classifier decide.

    Returns:
        Final state dict containing report_path and final_report.
    """
    from graph.financial_agent_graph import financial_agent

    print("\n" + "=" * 70)
    print("  AUTONOMOUS FINANCIAL REPORT AGENT")
    print("=" * 70)
    print(f"  Query : {query}")
    if report_type:
        print(f"  Report type (user-picked): {report_type}")
    print(f"  Graph recursion limit: {GRAPH_CONFIG['recursion_limit']}")
    print("=" * 70 + "\n")

    initial_state = {
        "raw_query":          query,
        "messages":           [],
        "ticker":             "",
        "elaborated_query":   "",
        "report_type":        report_type,
        "report_type_source": "",
        "roadmap":            {},
        "tool_calls_made":    [],
        "context_chunks":     [],
        "iteration":          0,
        "disabled_tools":     preseed_disabled_tools or [],
        "replan_count":       0,
        "needs_replan":       False,
        "replan_reason":      None,
        "replan_phase":       None,
        "rag_context":        "",
        "report_outline":     {},
        "final_report":       "",
        "report_path":        "",
    }

    if stream:
        print("[Streaming mode enabled – printing each node as it completes]\n")
        final_state = {}
        for chunk in financial_agent.stream(
            initial_state,
            config=GRAPH_CONFIG,        # ← recursion_limit passed here
        ):
            for node_name, node_output in chunk.items():
                print(f"\n{'─' * 50}")
                print(f"  ✓ NODE: {node_name}")
                print(f"{'─' * 50}")
                if "final_report" in node_output:
                    print(f"  Report: ~{len(node_output['final_report']):,} chars")
                if node_output.get("report_path"):
                    print(f"  Saved : {node_output['report_path']}")
                final_state.update(node_output)
        return final_state

    else:
        final_state = financial_agent.invoke(
            initial_state,
            config=GRAPH_CONFIG,        # ← recursion_limit passed here
        )

        print("\n" + "=" * 70)
        print("  AGENT RUN COMPLETE")
        print("=" * 70)
        print(f"  Ticker          : {final_state.get('ticker')}")
        print(f"  Tool calls made : {len(final_state.get('tool_calls_made', []))}")
        print(f"  Context chunks  : {len(final_state.get('context_chunks', []))}")
        print(f"  Report length   : ~{len(final_state.get('final_report', '')):,} chars")
        print(f"  Saved to        : {final_state.get('report_path')}")
        print("=" * 70 + "\n")

        return final_state


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Financial Report Agent (LangGraph + Groq)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=(
            "Perform a comprehensive financial analysis of Apple Inc (AAPL) "
            "for fiscal year 2024, including earnings analysis, risk assessment, "
            "and investment recommendation."
        ),
        help="Financial research query",
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Enable streaming mode (print each node as it completes)",
    )
    parser.add_argument(
        "--report-type", "-t",
        type=str,
        default="",
        choices=["", "COMPANY_RESEARCH", "COMPARATIVE_ANALYSIS",
                  "EARNINGS_EVENT", "VALUATION_RISK", "MACRO_COUNTRY", "OTHER"],
        help="Pre-select the report archetype. Leave blank to let the classifier decide.",
    )
    args = parser.parse_args()

    _setup_langsmith()
    optional_missing = check_env()
    preseed = [t for k in optional_missing for t in _KEY_TO_TOOLS.get(k, [])]
    run_agent(query=args.query, stream=args.stream,
              preseed_disabled_tools=preseed,
              report_type=args.report_type)


if __name__ == "__main__":
    main()
