"""
nodes/report_type_classifier.py – Runs after query_elaboration, before
roadmap_planner. Picks which report archetype this run produces.

Priority:
  1. If the user pre-selected a type (Streamlit sidebar / --report-type
     CLI flag), that wins unconditionally. Skip the LLM call.
  2. Otherwise, one LLM call classifies the elaborated query into one of
     config.REPORT_TYPES. The model is explicitly instructed to return
     OTHER when the query genuinely doesn't fit — otherwise the OTHER
     branch never fires and the classifier just always returns the
     closest of the 5.
  3. If both user and LLM return OTHER (or user left blank and LLM says
     OTHER), the roadmap planner will use config's OTHER default section
     list (EXEC_SUMMARY + CUSTOM_ANALYSIS) — the LLM populates
     CUSTOM_ANALYSIS with whatever the specific query needed.

This node does not implement the disagreement-confirm flow (that's the
next pass's LangGraph interrupt work). For now, on disagreement, user's
choice wins — same as when user pre-selected.
"""
from __future__ import annotations
import re
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from utils.groq_client import get_llm, invoke_with_retry
from config import (
    REPORT_TYPES, REPORT_TYPE_DESCRIPTIONS,
    COMPANY_RESEARCH, OTHER,
)

CLASSIFIER_SYSTEM_PROMPT = """You classify a financial research query into one
of the following report archetypes. Read the query, then output ONLY the
archetype name — no explanation, no punctuation, no markdown.

Archetypes:
""" + "\n".join(f"  {t}: {REPORT_TYPE_DESCRIPTIONS[t]}" for t in REPORT_TYPES) + """

EXAMPLES:
  "Analyse Apple AAPL for FY2024" → COMPANY_RESEARCH
  "Should I buy NVIDIA?" → COMPANY_RESEARCH
  "How is Tesla performing?" → COMPANY_RESEARCH
  "Compare Apple and Microsoft" → COMPARATIVE_ANALYSIS
  "Tesla vs Ford vs GM" → COMPARATIVE_ANALYSIS
  "Why did NVDA drop after earnings?" → EARNINGS_EVENT
  "Analyze Tesla's Q3 results" → EARNINGS_EVENT
  "Is Tesla overvalued?" → VALUATION_RISK
  "How risky is AAPL?" → VALUATION_RISK
  "Summarize today's market" → MACRO_COUNTRY
  "Compare India and China growth" → MACRO_COUNTRY
  "What's the impact of Fed rate changes on tech sector?" → MACRO_COUNTRY
  "Compute Apple's revenue CAGR and ROE" → COMPANY_RESEARCH
  "Explain Tesla's supply chain" → OTHER

RULES:
- Output exactly one of: """ + ", ".join(REPORT_TYPES) + """
- If a specific company ticker or name appears in the query, it is almost
  certainly NOT OTHER. Even narrow questions like "compute X's CAGR" are
  COMPANY_RESEARCH because they need company-level financial data.
- OTHER is ONLY for queries with no company AND no clear match to the other
  types. It should be rare — most financial queries fit one of the 5 types.
"""


def _extract_type(text: str) -> str:
    """Robustly pull a valid archetype label out of the model's response,
    tolerant of trailing punctuation / markdown / extra whitespace."""
    text = text.strip().upper()
    # Direct match first
    for t in REPORT_TYPES:
        if text == t:
            return t
    # Substring — pick the longest match to prefer "COMPARATIVE_ANALYSIS"
    # over the shorter "OTHER" if both somehow appear.
    hits = [t for t in REPORT_TYPES if re.search(rf"\b{t}\b", text)]
    if hits:
        return max(hits, key=len)
    return OTHER


def report_type_classifier_node(state: AgentState) -> dict:
    # User pre-selection wins if present and valid
    user_type = (state.get("report_type") or "").strip().upper()
    if user_type in REPORT_TYPES:
        print(f"[TypeClassifier] User pre-selected: {user_type} (skipping LLM classification)")
        return {
            "report_type": user_type,
            "report_type_source": "user",
        }

    # Otherwise, classify via LLM using the RAW query — short, clear signal.
    # The elaborated brief is too long and introduces noise (peer companies,
    # data requirements, etc.) that confused the classifier in earlier versions.
    llm = get_llm(temperature=0.0, max_tokens=32)
    messages = [
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=f"QUERY: {state.get('raw_query', '')}"),
    ]

    try:
        response = invoke_with_retry(llm, messages)
        classified = _extract_type(response.content)
    except Exception as exc:
        print(f"[TypeClassifier] LLM call failed ({exc}); falling back.")
        classified = COMPANY_RESEARCH if state.get("ticker") else OTHER

    print(f"[TypeClassifier] Classified as: {classified}")
    return {
        "report_type": classified,
        "report_type_source": "classifier",
    }
