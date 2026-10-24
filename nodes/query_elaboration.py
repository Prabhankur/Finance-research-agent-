"""
nodes/query_elaboration.py  –  Node 1: Elaborate the raw user query.
"""
from __future__ import annotations
import re
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from utils.groq_client import get_llm, invoke_with_retry

ELABORATION_SYSTEM_PROMPT = """You are a senior financial analyst and research director.
Your job is to take a user's raw financial query and elaborate it into a precise,
structured research brief that a team of analysts can execute.

AVAILABLE DATA SOURCES (be realistic about what can actually be fetched):
- yfinance: real-time price/multiples, income statement, balance sheet, cash flow (annual + quarterly)
- Alpha Vantage: earnings call transcripts, secondary financials source
- SEC EDGAR: 10-K, 10-Q, 8-K filings + XBRL structured numbers (US-listed companies only)
- Tavily: web search for news, analyst commentary, recent events
- FRED: US and global macro time-series (GDP, CPI, rates, unemployment, etc.)
- Calculator: derived metrics (CAGR, ROE, FCF Yield, Altman Z, PEG, etc.)

DO NOT ask for data these sources cannot provide, for example:
- Do NOT request "Bloomberg consensus" or "FactSet estimates" — we have neither.
  If analyst estimates are needed, phrase as "consensus estimates via web search."
- Do NOT request non-US company SEC filings — SEC EDGAR only covers US-listed.
- Do NOT request intraday price data or minute-level ticks — only daily/period data.
- Do NOT request internal management projections beyond what's in earnings transcripts.

Your response MUST begin with EXACTLY this first line, nothing before it:
TICKER: <SYMBOL>
Where <SYMBOL> is the primary company's stock ticker (infer it from the company
name if the user didn't give one, e.g. "Tesla" -> TSLA). Do not put any peer or
comparison company's ticker on this line — only the primary company being
analyzed. Follow that line with a blank line, then the rest of the brief.

Given the raw query, produce an elaborated version that clearly specifies:
1. PRIMARY COMPANY (if applicable): The exact company and ticker symbol being
   analyzed. Omit this field entirely for pure macro/market queries that don't
   have a primary company.
2. ANALYSIS SCOPE: What specific financial aspects need to be examined
   (earnings, risk, valuation, growth, competitive position, etc.). Be
   faithful to what the query actually asked — don't expand a narrow question
   into a full deep-dive.
3. TIME HORIZON: Which fiscal years / quarters are relevant.
4. COMPARISON CONTEXT: Are any peer companies or industry benchmarks needed?
   If you mention a peer company's ticker here, make clear it is a peer, not
   the primary company — the TICKER: line at the top is the only ticker that
   gets used to drive the actual research. Only include this section if the
   query is explicitly comparative or asks for peer benchmarks.
5. KEY QUESTIONS: The most important sub-questions this report must answer.
   Size this list to what the query actually needs — a narrow question may
   only need 2-3 sub-questions, a full deep-dive may need 5-7.
6. DATA REQUIREMENTS: What specific data points will be essential (financial
   statements, filings, transcripts, news, macro indicators, etc.).

Be thorough but concise. Output a structured brief, not a conversation."""


def query_elaboration_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.2)

    messages = [
        SystemMessage(content=ELABORATION_SYSTEM_PROMPT),
        HumanMessage(content=f"Raw query: {state['raw_query']}"),
    ]

    response = invoke_with_retry(llm, messages)
    elaborated = response.content

    # Extract ticker from the anchored first-line tag ONLY — not by
    # searching the whole free-form brief for the first "ticker"/"symbol"
    # mention. The old approach could — and did — grab a peer company's
    # ticker mentioned in the comparison-context section (e.g. asking about
    # Tesla, the brief discusses GM as a peer, and "GM" got extracted
    # instead of "TSLA" because it happened to appear first in the text).
    ticker_match = re.search(r"^\s*TICKER:\s*([A-Z]{1,5})\b", elaborated, re.IGNORECASE | re.MULTILINE)
    if not ticker_match:
        # Model didn't follow the format — fall back to the old
        # anywhere-in-text search, then the raw query, rather than failing.
        print("[Node 1] WARNING: No anchored TICKER: line found — falling back to loose search.")
        ticker_match = re.search(r"(?:ticker|symbol)[:\s]+([A-Z]{1,5})\b", elaborated, re.IGNORECASE)
    if not ticker_match:
        ticker_match = re.search(r"\(([A-Z]{1,5})\)", elaborated)
    if not ticker_match:
        ticker_match = re.search(r"\b([A-Z]{1,5})\b", state["raw_query"])

    ticker = ticker_match.group(1).upper() if ticker_match else "UNKNOWN"

    print(f"\n[Node 1] Query elaborated. Ticker extracted: {ticker}")
    print(f"[Node 1] Brief (first 300 chars):\n{elaborated[:300]}...\n")

    return {
        "elaborated_query": elaborated,
        "ticker":           ticker,
        "messages":         [response],
    }
