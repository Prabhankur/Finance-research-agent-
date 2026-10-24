"""
nodes/context_router.py – Routes accumulated context to report sections.

Replaces the old TF-IDF keyword-scoring rag_retriever. That approach scored
a chunk's relevance as (matching keywords / total tokens in chunk) — which
systematically penalises dense numeric/tabular data (a table of XBRL figures
has almost no natural-language overlap with a section query like "earnings
revenue EPS net income", even though it's exactly the data that section
needs), while text-heavy passages score better by comparison. That's a
structural reason financial numbers were losing to prose in the old design.

This version routes deterministically instead: every context chunk is
already tagged with which tool produced it (context_chunks[i]["tool"]), and
the roadmap planner already declares which tools each section depends on
(report_outline.sections[i]["data_sources"]). So for any section that
declared its data sources, we just pull those tool's successful chunks
directly — no scoring, no risk of numeric data losing a popularity contest.

Keyword scoring (same technique as before) is kept ONLY as a fallback, for:
  - sections that didn't declare data_sources (e.g. a fallback-roadmap
    section, or a section the LLM planner didn't tag)
  - chunks whose tool isn't referenced by any section's data_sources
    (e.g. a general web_search result that could apply to several sections)

Data gaps are surfaced explicitly: if a section depends on a tool that
failed or was disabled this run, a clear note is inserted into that
section's own context block, so the report generator can say "data
unavailable" honestly instead of silently missing it or guessing.
"""
from __future__ import annotations
import re
from collections import Counter
from graph.state import AgentState

SECTION_QUERIES: dict[str, str] = {
    # Shared
    "EXEC_SUMMARY": "executive summary revenue profit growth highlights key findings answer",
    "APPENDIX":     "financial table income statement balance sheet cash flow ratio data",

    # COMPANY_RESEARCH
    "COMPANY_PROFILE":     "company profile business overview sector industry products services employees",
    "EARNINGS_ANALYSIS":   "earnings revenue EPS net income profit margin quarterly annual growth YoY",
    "GROWTH_DRIVERS":      "growth driver product launch market expansion segment revenue growth",
    "VALUATION":           "valuation multiple PE ratio EV EBITDA DCF fair value price target",
    "RISK_ASSESSMENT":     "risk factors regulatory debt leverage litigation competition threats",
    "MANAGEMENT_GUIDANCE": "guidance outlook forecast future management commentary next quarter estimate",
    "INVESTMENT_THESIS":   "investment recommendation buy sell hold thesis target price conclusion",

    # COMPARATIVE_ANALYSIS
    "COMPANY_SNAPSHOTS":        "company snapshot overview market cap revenue employees business model",
    "FINANCIAL_COMPARISON":     "compare financial revenue margin EPS debt ROE cash flow",
    "GROWTH_COMPARISON":        "growth comparison CAGR revenue growth YoY segment growth",
    "VALUATION_COMPARISON":     "valuation compare PE EV EBITDA multiple peer",
    "RISK_COMPARISON":          "risk compare leverage competitive risk position",
    "PROFITABILITY_COMPARISON": "profit margin gross operating net ROE ROA ROIC compare",
    "WINNER_TRADEOFFS":         "winner tradeoff better investment growth value income",

    # EARNINGS_EVENT
    "EVENT_SUMMARY":          "event summary earnings release headline revenue EPS beat miss consensus",
    "MARKET_REACTION":        "stock price move volume analyst rating price target reaction",
    "COMPANY_FACTORS":        "revenue mix margin driver one-time item segment performance",
    "NEWS_CONTEXT":           "news context sector news macro peer results",
    "HISTORICAL_COMPARISON":  "prior quarter same year historical comparison trend inflection",
    "IMPACT_ASSESSMENT":      "impact assessment forward estimates investment case watch",

    # VALUATION_RISK
    "VALUATION_MODELS":       "valuation model PE EV EBITDA DCF fair value assumption",
    "PEER_MULTIPLES":         "peer multiple compare group PE EV EBITDA PB PEG",
    "SENSITIVITY_ANALYSIS":   "sensitivity growth rate margin discount fair value stress",
    "DOWNSIDE_SCENARIOS":     "downside bear case leverage liquidity risk catastrophic",
    "RISK_FACTORS":           "risk factor business financial market regulatory disclosure",
    "CONCLUSION":             "conclusion verdict buy sell risk analysis",

    # MACRO_COUNTRY
    "MACRO_INDICATORS":       "macro indicator GDP inflation CPI unemployment interest rate PMI",
    "MARKET_TRENDS":          "market trend equity index sector rotation bonds commodities FX",
    "SECTOR_PERFORMANCE":     "sector performance returns leadership rotation",
    "RATES_INFLATION_GDP":    "rates inflation GDP central bank policy stance expectations",
    "COUNTRY_COMPARISON":     "country compare economy growth inflation policy currency",
    "OUTLOOK":                "outlook base case near term watch data release central bank",

    # OTHER
    "CUSTOM_ANALYSIS":        "custom analysis specific question direct answer",
}

MAX_SECTION_CHARS = 6_000


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_chunk(chunk: str, query_tokens: list[str]) -> float:
    chunk_tokens = _tokenize(chunk)
    if not chunk_tokens:
        return 0.0
    counts = Counter(chunk_tokens)
    return sum(counts.get(t, 0) for t in query_tokens) / len(chunk_tokens)


def context_router_node(state: AgentState) -> dict:
    chunks = state.get("context_chunks", [])
    outline = state.get("report_outline", {})
    roadmap = state.get("roadmap", {})
    data_limitation_notes = roadmap.get("data_limitation_notes", [])

    if not chunks:
        print("[ContextRouter] WARNING: No context chunks to route.")
        return {"rag_context": "No data was collected during the research phase."}

    successful = [c for c in chunks if c.get("status") == "success"]
    failed_by_tool: dict[str, str] = {}
    for c in chunks:
        if c.get("status") in ("failed", "systemic_failure", "skipped"):
            failed_by_tool.setdefault(c["tool"], c.get("content", "unavailable"))

    by_tool: dict[str, list[str]] = {}
    for c in successful:
        by_tool.setdefault(c["tool"], []).append(str(c.get("content", "")))

    tools_claimed_by_sections: set[str] = set()
    sections = outline.get("sections", [])
    section_ids = [s["id"] for s in sections] if sections else list(SECTION_QUERIES.keys())
    section_by_id = {s["id"]: s for s in sections} if sections else {}

    for sec in sections:
        tools_claimed_by_sections.update(sec.get("data_sources", []))

    # Fallback keyword pool: chunks from tools no section explicitly claims.
    fallback_passages = [
        content for tool, contents in by_tool.items()
        if tool not in tools_claimed_by_sections
        for content in contents
    ]

    retrieved_sections: dict[str, str] = {}

    for sec_id in section_ids:
        sec = section_by_id.get(sec_id, {})
        data_sources = sec.get("data_sources", [])

        parts: list[str] = []
        gap_notes: list[str] = []

        if data_sources:
            # Deterministic routing: pull this section's declared sources directly.
            for tool in data_sources:
                if tool in by_tool:
                    parts.extend(by_tool[tool])
                elif tool in failed_by_tool:
                    gap_notes.append(f"[Data unavailable] {tool}: {failed_by_tool[tool]}")
            # Also sweep in any keyword-relevant fallback passages, in case a
            # generic tool (e.g. web_search) is relevant but wasn't declared.
            if fallback_passages:
                q_tokens = _tokenize(SECTION_QUERIES.get(sec_id, sec_id.lower()))
                scored = sorted(
                    ((p, _score_chunk(p, q_tokens)) for p in fallback_passages),
                    key=lambda x: x[1], reverse=True,
                )
                parts.extend(p for p, score in scored[:3] if score > 0)
        else:
            # No metadata for this section at all — pure keyword fallback
            # over every successful chunk, same technique as the old node.
            q_tokens = _tokenize(SECTION_QUERIES.get(sec_id, sec_id.lower()))
            all_text = [content for contents in by_tool.values() for content in contents]
            scored = sorted(
                ((p, _score_chunk(p, q_tokens)) for p in all_text),
                key=lambda x: x[1], reverse=True,
            )
            parts.extend(p for p, _ in scored[:6])

        section_text = "\n\n".join(gap_notes + parts)[:MAX_SECTION_CHARS]
        retrieved_sections[sec_id] = section_text
        print(f"  [ContextRouter] [{sec_id}] {len(parts)} source blocks, "
              f"{len(gap_notes)} gap notes ({len(section_text):,} chars)")

    rag_lines = ["=" * 70, "RETRIEVED CONTEXT BY REPORT SECTION", "=" * 70]
    if data_limitation_notes:
        rag_lines.append("\n--- RUN-WIDE DATA LIMITATIONS ---")
        rag_lines.extend(f"- {n}" for n in data_limitation_notes)
    for sec_id, text in retrieved_sections.items():
        rag_lines.append(f"\n--- SECTION {sec_id} ---\n{text}")

    rag_context = "\n".join(rag_lines)
    print(f"\n[ContextRouter] Routing complete. Total context: ~{len(rag_context):,} chars.")
    return {"rag_context": rag_context}
