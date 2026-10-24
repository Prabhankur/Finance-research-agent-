"""
nodes/roadmap_planner.py — Creates the research plan + report outline,
using the classified report_type to bias which sections and tools are used.

The type gives the planner a starting section list (from
config.DEFAULT_SECTIONS_BY_TYPE). The LLM adjusts that list based on the
specific query — dropping sections that clearly don't apply, keeping the
ones that do. Section IDs it emits MUST come from the SECTION_INSTRUCTIONS
dict in report_generator.py; the planner isn't allowed to invent new
section names (except CUSTOM_ANALYSIS, which the OTHER type uses).
"""
from __future__ import annotations
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from tools import TOOL_DESCRIPTIONS
from utils.groq_client import get_llm, invoke_with_retry
from config import (
    REPORT_TYPES, REPORT_TYPE_LABELS, DEFAULT_SECTIONS_BY_TYPE,
    COMPANY_RESEARCH, OTHER,
)
from nodes.report_generator import SECTION_INSTRUCTIONS


def _build_system_prompt() -> str:
    """Assembled once at import so we don't recompute the section catalog
    on every call. Includes the full section library so the LLM knows what
    IDs it's allowed to emit."""
    section_catalog_lines = []
    for sec_id, instr in SECTION_INSTRUCTIONS.items():
        # First line of instruction is usually "Write the X section covering:"
        # — pull it as a one-line description so the catalog stays compact.
        first_line = instr.strip().split("\n")[0].rstrip(":")
        section_catalog_lines.append(f"  {sec_id}: {first_line}")
    section_catalog = "\n".join(section_catalog_lines)

    defaults_lines = []
    for rt in REPORT_TYPES:
        default_secs = DEFAULT_SECTIONS_BY_TYPE.get(rt, [])
        defaults_lines.append(f"  {rt}: {', '.join(default_secs)}")
    defaults_block = "\n".join(defaults_lines)

    from utils.chart_generator import CHART_TYPE_DESCRIPTIONS

    return f"""You are a senior financial research strategist. Given the elaborated
research brief and the classified REPORT TYPE, produce THREE things:
  1. A research plan (which tools to call, in what order, with what args)
  2. A report outline (which sections to write)
  3. A chart plan (which charts to embed in which sections)

{TOOL_DESCRIPTIONS}

═══════════════════════════════════════════════════════════════════════════
CHART LIBRARY — charts the report can include
═══════════════════════════════════════════════════════════════════════════
{CHART_TYPE_DESCRIPTIONS}
Charts are generated automatically from fetched data. You just specify
which chart_type goes in which section. If the data isn't available at
generation time, the chart is silently skipped — so plan optimistically.

═══════════════════════════════════════════════════════════════════════════
SECTION LIBRARY — the ONLY section IDs you may emit
═══════════════════════════════════════════════════════════════════════════
{section_catalog}

═══════════════════════════════════════════════════════════════════════════
DEFAULT SECTIONS PER REPORT TYPE — your starting point, not a hard rule
═══════════════════════════════════════════════════════════════════════════
{defaults_block}

═══════════════════════════════════════════════════════════════════════════
HOW TO BUILD THE OUTLINE
═══════════════════════════════════════════════════════════════════════════
1. Take the default section list for the given REPORT TYPE as your starting point.
2. Read the specific query and SIZE the section list to what it actually needs:

   BROAD QUERIES ("Analyse Apple for FY2024", "Should I buy NVIDIA?"):
   → Keep most of the default sections (6-9 sections). These queries want
     a full picture. Don't drop sections just because the query didn't
     explicitly name them — "analyse X" implies earnings, risk, valuation.

   NARROW QUERIES ("What was Apple's CAGR last 3 years?", "Compute ROE"):
   → Reduce to 3-5 sections. Keep EXEC_SUMMARY (always), the section that
     directly answers the question, APPENDIX if there's raw data to show.
     Drop everything the user didn't ask about.

   OTHER TYPE QUERIES:
   → You have the FULL section library as your menu. Pick whichever
     sections genuinely fit what the query is asking about — could be 3,
     could be 10, depends on the query. Always include EXEC_SUMMARY.
     Include CUSTOM_ANALYSIS to directly address the specific angle.
     Pick additional sections from the library that are relevant.

3. You can also:
   - REMOVE sections the query explicitly says to skip ("but don't include risks").
   - ADD sections from the library if the query calls for them
     ("also give me a valuation view" on a research query → add VALUATION).
4. Every section MUST have a "data_sources" list naming which tools its
   content comes from — e.g. ["financial_data_api", "sec_filing_search"].
   If a section is genuinely synthesis-only (INVESTMENT_THESIS,
   WINNER_TRADEOFFS, IMPACT_ASSESSMENT), an empty list is fine.
4. For the OTHER report type, use EXEC_SUMMARY + CUSTOM_ANALYSIS as the base
   and ADD whichever other library sections apply. CUSTOM_ANALYSIS is where
   you address the specific angle the query focused on.

═══════════════════════════════════════════════════════════════════════════
HOW TO BUILD THE RESEARCH PLAN
═══════════════════════════════════════════════════════════════════════════
Size the plan to the query. The agent parallelizes by tool group (SEC,
web search, and FRED run simultaneously alongside AV calls), but Alpha
Vantage calls (financial_data_api + earnings_transcript) are rate-limited
to ~5/min, so EVERY AV CALL COSTS ~13 SECONDS. Minimize AV calls.

STEP BUDGETS BY REPORT TYPE (these are ceilings, not targets):
  COMPANY_RESEARCH:      6-10 steps (1 financial_data_api, 1-2 SEC, 1 transcript, 1-2 web, 1-3 calculator)
  COMPARATIVE_ANALYSIS:  8-14 steps (1 financial_data_api PER company, 1 SEC per company, 1 transcript per company, 2-3 web, 1-3 calculator)
  EARNINGS_EVENT:        5-8 steps (1 financial_data_api, 1 transcript, 2-3 web)
  VALUATION_RISK:        6-10 steps (1 financial_data_api, 1-2 SEC, 1-2 web, 2-4 calculator)
  MACRO_COUNTRY:         5-8 steps (2-4 fred_series, 2-3 web, no SEC/financial_data)

CRITICAL RULES for AV calls:
- ONE financial_data_api(statement_type="all") per company gives you EVERYTHING
  (overview, income, balance, cashflow, ratios). NEVER call it twice for the
  same company with different statement_types — "all" already includes them.
- ONE earnings_transcript per company (most recent quarter). Do NOT fetch
  4 quarters of transcripts — that's 4 AV calls (52 seconds of waiting)
  for marginal value. One recent transcript gives you current guidance.
- For a 2-company comparison: that's 2 financial_data_api + 2 transcripts
  = 4 AV calls total. NOT 8 financial_data_api + 8 transcripts = 16.

Other guidelines:
- ONE company in query → one financial_data_api call, not five.
- Comparison of N named companies → one financial_data_api call per company.
- Do NOT auto-generate peer fetches for a single-company query.
- Only include sec_filing_search when regulatory content is actually needed.
  One 10-K per company is usually enough — not 3 years of 10-Ks each.
- Only include fred_series when macro data is relevant.
  relevant — MACRO_COUNTRY and sometimes EARNINGS_EVENT queries need it,
  single-stock analysis usually doesn't.
- Only include calculator steps for metrics the report will actually cite.
- For calculator steps, args can be a placeholder (e.g. {{"metric": "cagr"}}).
  Make the "purpose" field specific — say which numbers it needs and where
  they come from (e.g. "CAGR of revenue using FY2022 and FY2024 figures from step 1").

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════
Respond ONLY with valid JSON, no markdown fences. Structure:

{{
  "research_plan": [
    {{"step": 1, "tool": "financial_data_api",
      "args": {{"ticker": "AAPL", "statement_type": "all", "period": "annual", "years": 3}},
      "purpose": "Get full financial overview"}},
    {{"step": 5, "tool": "calculator",
      "args": {{"metric": "cagr"}},
      "purpose": "CAGR of revenue using FY2022 and FY2024 figures from step 1"}}
  ],
  "report_outline": {{
    "title": "...",
    "sections": [
      {{"id": "EXEC_SUMMARY", "title": "Executive Summary",
        "data_sources": ["financial_data_api"]}},
      {{"id": "EARNINGS_ANALYSIS", "title": "Earnings Analysis",
        "data_sources": ["financial_data_api", "earnings_transcript"]}}
    ],
    "charts": [
      {{"chart_type": "revenue_trend", "section": "EARNINGS_ANALYSIS"}},
      {{"chart_type": "margin_trend", "section": "EARNINGS_ANALYSIS"}},
      {{"chart_type": "valuation_multiples", "section": "VALUATION"}},
      {{"chart_type": "price_position", "section": "INVESTMENT_THESIS"}}
    ]
  }}
}}
"""


ROADMAP_SYSTEM_PROMPT = _build_system_prompt()


def roadmap_planner_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.1, max_tokens=4096)
    report_type = state.get("report_type") or COMPANY_RESEARCH
    default_sections = DEFAULT_SECTIONS_BY_TYPE.get(report_type, DEFAULT_SECTIONS_BY_TYPE[COMPANY_RESEARCH])

    user_content = (
        f"REPORT TYPE: {report_type} ({REPORT_TYPE_LABELS.get(report_type, report_type)})\n"
        f"DEFAULT SECTION LIST FOR THIS TYPE: {', '.join(default_sections)}\n"
        f"TICKER: {state.get('ticker', '(none)')}\n\n"
        f"ELABORATED RESEARCH BRIEF:\n{state.get('elaborated_query', state.get('raw_query', ''))}"
    )

    messages = [
        SystemMessage(content=ROADMAP_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = invoke_with_retry(llm, messages)
    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    try:
        roadmap = json.loads(raw)
    except json.JSONDecodeError:
        print("[RoadmapPlanner] WARNING: Could not parse JSON. Using type-default fallback.")
        roadmap = _fallback_roadmap(state.get("ticker", ""), report_type)

    # Post-validation: any section ID the LLM emitted that isn't in the
    # library gets dropped, since report_generator wouldn't know how to
    # write it. This is a safety net — the system prompt already tells the
    # LLM to only use IDs from the library, but LLMs sometimes reach.
    outline = roadmap.get("report_outline", {})
    sections = outline.get("sections", [])
    valid_ids = set(SECTION_INSTRUCTIONS.keys())
    kept_sections = []
    dropped_sections = []
    for sec in sections:
        if sec.get("id") in valid_ids:
            kept_sections.append(sec)
        else:
            dropped_sections.append(sec.get("id"))
    if dropped_sections:
        print(f"[RoadmapPlanner] Dropped unknown section IDs: {dropped_sections}")
    outline["sections"] = kept_sections
    roadmap["report_outline"] = outline

    plan = roadmap.get("research_plan", [])
    print(f"\n[RoadmapPlanner] Type={report_type}  Plan steps={len(plan)}  Sections={len(kept_sections)}")

    return {
        "roadmap":         roadmap,
        "report_outline":  outline,
        "messages":        [response],
        "iteration":       0,
        "tool_calls_made": [],
        "context_chunks":  [],
        "disabled_tools":  [],
        "replan_count":    0,
        "needs_replan":    False,
        "replan_reason":   None,
        "replan_phase":    None,
    }


def _fallback_roadmap(ticker: str, report_type: str) -> dict:
    """Used only when the LLM's own JSON response failed to parse. Produces
    a minimally-useful plan and an outline built from the type's default
    section list."""
    default_sections = DEFAULT_SECTIONS_BY_TYPE.get(report_type, DEFAULT_SECTIONS_BY_TYPE[COMPANY_RESEARCH])
    sections = [{"id": s, "title": s.replace("_", " ").title(), "data_sources": []}
                for s in default_sections]

    if ticker:
        research_plan = [
            {"step": 1, "tool": "financial_data_api",
             "args": {"ticker": ticker, "statement_type": "all", "period": "annual", "years": 3},
             "purpose": "Full financial overview (fallback plan)"},
            {"step": 2, "tool": "web_search",
             "args": {"query": f"{ticker} latest news analysis", "num_results": 8},
             "purpose": "Recent news and analysis"},
        ]
    else:
        # No ticker — probably a MACRO_COUNTRY or OTHER query
        research_plan = [
            {"step": 1, "tool": "web_search",
             "args": {"query": "current market macro trends analysis", "num_results": 10},
             "purpose": "General market and macro context (fallback plan)"},
        ]

    return {
        "research_plan": research_plan,
        "report_outline": {
            "title": f"Financial Report – {ticker or 'General'}",
            "sections": sections,
        },
    }
