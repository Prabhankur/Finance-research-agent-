"""
nodes/report_generator.py — Generates the final report section by section.

The section library below is the full menu across all supported report types.
Which sections get used in any given run is decided upstream by roadmap_planner
(based on the report_type set by report_type_classifier), NOT by this file.
This file just knows how to WRITE any section it's asked to.

Adding a new section anywhere:
  1. Add its ID + writing instruction to SECTION_INSTRUCTIONS below.
  2. Add its ID to at least one type's default list in config.DEFAULT_SECTIONS_BY_TYPE.
  3. Add a keyword-fallback query for it in nodes/context_router.py's SECTION_QUERIES.
"""
from __future__ import annotations
import json
import os
from datetime import date
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from utils.groq_client import get_llm, invoke_with_retry


# ─────────────────────────────────────────────────────────────────────────────
#  Section library — every possible section any report type can use.
# ─────────────────────────────────────────────────────────────────────────────

SECTION_INSTRUCTIONS: dict[str, str] = {
    # ── Shared ────────────────────────────────────────────────────────────────
    "EXEC_SUMMARY": """Write the Executive Summary covering:
- What the query asked and the report's headline answer
- Top 3 data points that support that answer (with real numbers)
- Key risks or caveats
- Bottom line stance (one sentence)
Format: 5-row summary table (Metric | Value | Notes) plus a short paragraph.""",

    "APPENDIX": """Write an Appendix with the raw financial tables the report drew from:
TABLE: Full Income Statement (3 years annual, if available)
TABLE: Full Balance Sheet (latest 2 years, if available)
TABLE: Cash Flow Summary (3 years)
TABLE: Key Ratios (latest year)
Use actual numbers from context. Mark as 'N/A' if not available.""",

    # ── COMPANY_RESEARCH type ────────────────────────────────────────────────
    "COMPANY_PROFILE": """Write the Company Profile using ONLY what the research data actually contains.

DO NOT write tables with N/A values. If a specific piece of information isn't in the context, skip it entirely — a missing table is better than a table of N/As.

Write what you DO have:
- If Market Overview is present: write the business description, sector, industry, current price, market cap
- If SEC 10-K is present: write the business overview, products/services, geographic footprint, segments
- If web search results mention executives, employees, headquarters: include those
- Only write a "Key Company Facts" table if you have at least 3 real values to put in it
- Only write a "Revenue by Segment" table if segment data actually exists in the context
- Skip "Historical Milestones" and "Key Executives" tables entirely if those fields are absent — write a sentence saying that information wasn't in the fetched data sources, then move on

The goal is a concise, honest Company Profile — not a table-heavy section with mostly N/A.""",

    "EARNINGS_ANALYSIS": """Write the Earnings Analysis using ONLY numbers that appear in the research data.

DO NOT write tables with N/A values. Only include a table if you have real numbers to put in at least 3 of its cells.

Use what you actually have:
- Income Statement data: write revenue, gross profit, operating income, net income for available years
- Market Overview: quarterly revenue growth, profit margin, EPS
- SEC XBRL data: use those numbers for the most recent fiscal year
- Earnings transcript: management commentary, guidance, beat/miss vs estimates
- Web search: analyst estimates, quarterly results mentioned in news

If only 2 years of annual data exist (not 3), write 2 years — don't add an N/A column for the missing year.
If quarterly data exists, write a quarterly tracker with real numbers only.
Skip any sub-section where you genuinely have no data — do not pad with N/A rows.""",

    "GROWTH_DRIVERS": """Write the Growth Drivers section covering:
- Primary revenue growth engines (products, geographies, segments)
- Quantified growth assumptions where possible
- New product launches / market expansions relevant to the outlook
- Structural headwinds
Distinguish clearly between historical growth drivers and forward-looking ones.""",

    "VALUATION": """Write the Valuation section covering:
- Current multiples: P/E, EV/EBITDA, P/B, PEG (whichever the context provides)
- How these compare to sector and peer averages
- A brief DCF or multiple-based fair-value range if enough data is present
- Whether the stock looks cheap, fair, or expensive on the numbers as of the report date""",

    "RISK_ASSESSMENT": """Write the Risk Assessment covering:
- Top 5 market risks (macro, competition, pricing pressure)
- Operational risks (supply chain, execution, key-person)
- Regulatory & legal risks
- Financial risks (debt load, refinancing, covenant risk, liquidity)
- Balance sheet stress indicators (D/E, interest coverage, current ratio)
TABLE IF DATA EXISTS: Risk Matrix (Risk | Likelihood | Impact | Mitigation) — only write this if you have at least 4 specific real risks from the context""",

    "MANAGEMENT_GUIDANCE": """Write the Management Guidance & Forward Outlook section covering:
- Official management guidance for next quarter/year (revenue, EPS, margins)
- Key assumptions management is making
- Analyst consensus estimates vs management guidance
- Catalysts for upside or downside surprises
TABLE IF DATA EXISTS: Forward Estimates — only write this table if the transcript or web search contains actual forward guidance numbers""",

    "INVESTMENT_THESIS": """Write the Investment Thesis & Recommendation covering:
- Clear BUY / HOLD / SELL stance with rationale
- 3 bull case arguments (with supporting data points)
- 3 bear case arguments (with supporting data points)
- Price target rationale (P/E or EV/EBITDA multiple applied to estimates)
- Time horizon and key risks to the thesis""",

    # ── COMPARATIVE_ANALYSIS type ────────────────────────────────────────────
    "COMPANY_SNAPSHOTS": """Write a Company Snapshots section giving a one-paragraph
overview of each company being compared. For each company include: what they do,
their scale (revenue, market cap), and their positioning in the market.
TABLE IF DATA EXISTS: Snapshot Table — only include fields you have real values for""",

    "FINANCIAL_COMPARISON": """Write a Financial Comparison covering the same
metrics side-by-side for every company in scope. Cover: revenue, gross/op/net
margin, EPS, debt/equity, ROE, FCF.
TABLE IF DATA EXISTS: Financial Comparison Matrix — only include rows where you have real values for all companies
Be strictly evenhanded — do not favor one company in the narrative.""",

    "GROWTH_COMPARISON": """Write a Growth Comparison section covering:
- 3-year revenue CAGR per company
- Recent quarterly YoY growth per company
- Segment-level growth if available
TABLE IF DATA EXISTS: Growth Comparison — skip rows or columns where data is missing""",

    "VALUATION_COMPARISON": """Write a Valuation Comparison covering the same
multiples for every company in scope: P/E, forward P/E, EV/EBITDA, P/B, PEG.
TABLE IF DATA EXISTS: Valuation Multiples Comparison — only write if you have multiples for at least 2 companies
Discuss which company is cheaper or more expensive on the numbers and why.""",

    "RISK_COMPARISON": """Write a Risk Comparison section covering financial
leverage, competitive positioning, and specific company risks side-by-side.
TABLE IF DATA EXISTS: Risk Comparison — only write if you have specific risks for each company""",

    "PROFITABILITY_COMPARISON": """Write a Profitability Comparison section covering:
- Gross margin, operating margin, net margin per company
- Return metrics (ROE, ROA, ROIC where available)
- Trend direction over the last 3 years
TABLE IF DATA EXISTS: Profitability Metrics — only include metrics you have data for""",

    "WINNER_TRADEOFFS": """Write a Winner / Trade-offs conclusion covering:
- Which company screens better on the financial numbers, and why
- What each company does better than the other (trade-offs)
- Which one would suit which type of investor (growth vs value vs income)
- The honest answer to "which is the better investment right now"
Be direct — comparative reports exist to make comparisons, not hedge them.""",

    # ── EARNINGS_EVENT type ──────────────────────────────────────────────────
    "EVENT_SUMMARY": """Write an Event Summary covering:
- What happened (release / event / price move) and when
- Headline numbers from the release: revenue, EPS, guidance changes
- Whether it beat, missed, or matched consensus
- Key one-sentence narrative from the release
TABLE IF DATA EXISTS: Headline Numbers — only write if you have both reported and consensus numbers""",

    "MARKET_REACTION": """Write a Market Reaction section covering:
- Stock's price move on the release / event (day, week)
- Volume vs average
- Analyst rating and price-target changes in response
- Sector or peer moves for context""",

    "COMPANY_FACTORS": """Write a Company Factors section explaining what
INSIDE the company drove the result. Cover: revenue mix changes, margin
drivers, one-time items, segment performance. Distinguish operating from
non-operating.""",

    "NEWS_CONTEXT": """Write a News Context section covering the outside events,
sector news, macro moves, or peer results that shaped the quarter/event.
Cite recent news items from the research context — do not invent citations.""",

    "HISTORICAL_COMPARISON": """Write a Historical Comparison covering how this
event/quarter compares to prior periods: same quarter last year, previous
quarter, prior guidance history. Highlight whether this is an inflection or
continuation of a trend.""",

    "IMPACT_ASSESSMENT": """Write an Impact Assessment covering:
- What this event changes about the investment case
- Whether the forward estimates should move up, down, or stay
- What to watch next
- One-sentence updated stance""",

    # ── VALUATION_RISK type ──────────────────────────────────────────────────
    "VALUATION_MODELS": """Write a Valuation Models section covering EACH
valuation approach the context supports:
- P/E vs. history and sector
- EV/EBITDA vs. peers
- DCF fair value range (if enough cash-flow data is present)
- Precedent transactions (if any)
For each, state the assumption, the number, and the resulting fair value.
TABLE IF DATA EXISTS: Valuation Summary — only write methods you can actually support with data from context""",

    "PEER_MULTIPLES": """Write a Peer Multiples section comparing this company's
current multiples to a peer group. Discuss any multiple gap vs. peers and
whether it is justified by growth, margins, or risk.
TABLE IF DATA EXISTS: Peer Multiples — only include companies and metrics where you have real values""",

    "SENSITIVITY_ANALYSIS": """Write a Sensitivity Analysis section stress-testing
the base-case valuation to changes in the key drivers: revenue growth,
margins, discount rate.
TABLE IF DATA EXISTS: Sensitivity Grid — ONLY write this if you have real base-case revenue growth and margin figures from the context. Do NOT invent discount rates or growth assumptions.
Show the range fair value can plausibly take, not just a point estimate.""",

    "DOWNSIDE_SCENARIOS": """Write a Downside Scenarios section covering:
- Realistic bear case: what does the stock do if key assumptions break?
- Catastrophic case: leverage / liquidity risk if applicable
- What the downside implies for position sizing
Do not sugarcoat — this section exists specifically to be honest about risk.""",

    "RISK_FACTORS": """Write a Risk Factors section drawing from the company's own
disclosures and current context. Group into: business, financial, market,
regulatory. Cite specifically where each risk comes from.""",

    "CONCLUSION": """Write a Conclusion tying the valuation, sensitivity, and risk
analyses back into a single verdict: is the stock a buy at current levels
given the risks? Under what conditions would that change? Be direct.""",

    # ── MACRO_COUNTRY type ───────────────────────────────────────────────────
    "MACRO_INDICATORS": """Write a Macro Indicators section covering the key
economic indicators relevant to the query: GDP growth, inflation (CPI/PCE),
unemployment, interest rates, PMI. Use FRED numbers if present in the context.
TABLE IF DATA EXISTS: Macro Indicators — only include indicators present in FRED or web search data""",

    "MARKET_TRENDS": """Write a Market Trends section covering equity index
performance, sector rotation, and any notable moves in bonds/commodities/FX
relevant to the query. Frame as a narrative about what's driving markets
right now, not just numbers.""",

    "SECTOR_PERFORMANCE": """Write a Sector Performance section covering how the
sectors mentioned in the query have performed recently, what's driving that
performance, and how leadership has rotated.
TABLE IF DATA EXISTS: Sector Returns — only write if web search returned actual sector return data""",

    "RATES_INFLATION_GDP": """Write a Rates, Inflation, and GDP section going
deeper on those three variables specifically: current levels, direction,
central bank stance, and near-term expectations. Use FRED data where the
context provides it.""",

    "COUNTRY_COMPARISON": """Write a Country Comparison section if the query
compared two or more countries or economies. Cover: growth, inflation,
policy stance, currency, market performance.
TABLE IF DATA EXISTS: Country Comparison — only write if FRED or web data contains figures for the specific countries asked about
If the query didn't ask for country comparison, note that briefly and skip.""",

    "OUTLOOK": """Write an Outlook section covering:
- Base case for the near-term (next 3-6 months)
- What could change the picture in either direction
- What to watch (next data releases, central bank meetings, etc.)""",

    # ── OTHER type (custom queries the archetypes don't cover) ────────────────
    "CUSTOM_ANALYSIS": """Write a Custom Analysis section that directly addresses
the specific angle the user's original query focused on. The query didn't
fit any of the standard report archetypes cleanly, so this section IS the
main content of the report.

Use the research data provided to answer the actual question asked, in
whatever structure makes sense for that question. This is the only section
in this report where you have real latitude on structure — use it to give
the user a genuine, direct answer to what they asked, not a template.

Still follow the writing standards: real numbers only, no fabrication,
markdown tables where they clarify, direct tone. If the data doesn't cover
some part of what was asked, say so plainly.""",
}


# ─────────────────────────────────────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────────────────────────────────────

REPORT_SYSTEM_PROMPT = """You are a senior financial analyst writing a professional investment research report.

WRITING STANDARDS:
- Write in a professional, analytical tone — like a Goldman Sachs or Morgan Stanley research report.
- Use ACTUAL numbers from the provided context. Never fabricate data.
- If a number is not in the context, write 'N/A' or 'data unavailable' — never guess.
- **DO NOT write tables where most cells are N/A.** A table that is mostly empty adds no value
  and makes the report look broken. If fewer than half of a table's cells have real values,
  skip the table entirely and write a single sentence noting what data is unavailable.
- **Adapt section length to available data.** If only 2 years of income statement data exist,
  write a 2-year table — not a 3-year table with an N/A column. If executive names aren't in
  the context, skip the executives table entirely rather than filling it with N/A rows.
- Keep sections concise. A well-written 200-word section with real data beats a 600-word
  section padded with N/A tables and apologies for missing information.
- Never name a data provider, vendor, or database that isn't explicitly named
  in the source text. Prior versions of this agent were caught inventing
  provider names ("per FMP", "according to Refinitiv") for numbers the tool
  actually pulled from a different source. If you want to cite the source of
  a number, either quote the exact tag it appeared under (e.g. "[FinData]",
  "[SEC XBRL]", "[WebSearch]", "[FRED]") or don't attribute it at all.
- If the context includes a line starting with '[Data unavailable]', that means a data
  source failed or was unavailable this run. Don't ignore it and don't quietly work
  around it — state plainly which specific data point is unavailable and why, then
  continue with whatever data you do have.
- If a section's routed context is empty because its data_sources list was
  intentionally left empty by the planner (meaning: this section wasn't relevant
  to the specific query), write a brief 1-2 sentence note explaining it isn't
  applicable to this query and move on. Do not invent content to fill the section.
- Format all tables using Markdown pipe syntax (| Col | Col |).
- Use ## for section headers, ### for subsection headers.
- Bold key metrics and conclusions.
- Each table must have a header row and a separator row (|---|---|).
- Be concise but thorough — quality over padding. Write to the depth the
  available data supports; there is no fixed length target."""


# ─────────────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────────────

def _write_section(llm, section_id, section_title, context, ticker, roadmap_summary):
    instruction = SECTION_INSTRUCTIONS.get(
        section_id,
        f"Write the {section_title} section addressing what's relevant to the query, "
        f"using the research data provided. Follow the general writing standards."
    )
    messages = [
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"REPORT CONTEXT: {ticker or 'N/A'}\n\n"
            f"SECTION TO WRITE: {section_title} ({section_id})\n\n"
            f"INSTRUCTIONS:\n{instruction}\n\n"
            f"RESEARCH DATA:\n{context}\n\n"
            f"ROADMAP NOTES:\n{roadmap_summary}\n\n"
            f"Write this section now using markdown formatting."
        )),
    ]
    response = invoke_with_retry(llm, messages, max_retries=5)
    return response.content


def report_generator_node(state: AgentState) -> dict:
    llm     = get_llm(temperature=0.3, max_tokens=4096)
    ticker  = state.get("ticker", "")
    outline = state.get("report_outline", {})
    rag_ctx = state.get("rag_context", "")
    report_type = state.get("report_type", "COMPANY_RESEARCH")

    # Parse RAG context into per-section dict
    section_contexts: dict[str, str] = {}
    current_section = "GENERAL"
    buffer: list[str] = []
    for line in rag_ctx.split("\n"):
        if line.startswith("--- SECTION "):
            if buffer:
                section_contexts[current_section] = "\n".join(buffer)
                buffer = []
            current_section = line.replace("--- SECTION ", "").replace(" ---", "").strip()
        else:
            buffer.append(line)
    if buffer:
        section_contexts[current_section] = "\n".join(buffer)

    roadmap = state.get("roadmap", {})
    roadmap_summary = json.dumps(roadmap.get("report_outline", {}), indent=2)[:1500]

    sections = outline.get("sections", [])
    if not sections:
        # Absolute last-ditch fallback — planner failed to produce a section
        # list at all. Give the user *something* rather than crashing.
        sections = [
            {"id": "EXEC_SUMMARY", "title": "Executive Summary"},
            {"id": "CUSTOM_ANALYSIS", "title": "Analysis"},
        ]

    today = date.today().strftime("%B %d, %Y")
    report_title = outline.get("title") or (
        f"Financial Analysis – {ticker}" if ticker else "Financial Research Report"
    )

    header_line = f"**Date:** {today}"
    if ticker:
        header_line += f"  |  **Ticker:** {ticker}"
    header_line += f"  |  **Report Type:** {report_type}  |  **Prepared by:** Financial Analysis Agent"

    report_parts = [
        f"# {report_title}",
        header_line,
        "---",
        "> *This report was generated by an autonomous AI financial research agent.*",
        "> *All data is sourced from public filings, financial APIs, and verified news.*",
        "---",
        "## Table of Contents",
    ]
    for i, sec in enumerate(sections, 1):
        report_parts.append(f"{i}. {sec.get('title', sec['id'])}")
    report_parts.append("\n---\n")

    # Build a lookup: section_id → set of tool names that actually contributed data
    section_source_tools: dict[str, set] = {}
    context_chunks = state.get("context_chunks", [])
    for sec in sections:
        sec_id = sec["id"]
        declared = set(sec.get("data_sources", []))
        # Which of those declared tools actually returned successful data?
        contributing = set()
        for chunk in context_chunks:
            if chunk.get("status") == "success" and chunk.get("tool") in declared:
                contributing.add(chunk["tool"])
        section_source_tools[sec_id] = contributing

    # Get planner's chart specs (if any)
    planned_charts = outline.get("charts", [])
    if planned_charts:
        print(f"[ReportGen] Planner specified {len(planned_charts)} charts.")

    # Pre-generate all charts so they're ready for inline embedding
    from pathlib import Path
    chart_output_dir = str(Path("output"))
    try:
        from utils.chart_generator import generate_charts_for_section
        charts_available = True
    except ImportError:
        charts_available = False

    total_sections = len(sections)
    section_chart_paths: list[str] = []

    for idx, sec in enumerate(sections, 1):
        sec_id    = sec["id"]
        sec_title = sec.get("title", sec_id)
        print(f"[ReportGen] Writing section {idx}/{total_sections}: {sec_id} – {sec_title}")
        ctx = section_contexts.get(sec_id, "")
        if not ctx:
            ctx = rag_ctx[:3000]
        if len(ctx) > 4000:
            ctx = ctx[:4000] + "\n[...context truncated to fit model context window]"
        section_text = _write_section(llm, sec_id, sec_title, ctx, ticker, roadmap_summary)

        # Append deterministic source tag
        sources = section_source_tools.get(sec_id, set())
        if sources:
            source_tag = ", ".join(f"[{s}]" for s in sorted(sources))
            section_text += f"\n\n*Sources for this section: {source_tag}*"
        else:
            section_text += "\n\n*Sources for this section: synthesis from earlier sections — no direct tool data routed here.*"

        report_parts.append(f"\n---\n\n## {sec_title}\n\n{section_text}")

        # Embed charts INLINE — uses planner specs if available, fallback otherwise
        if charts_available and ticker:
            try:
                sec_charts = generate_charts_for_section(
                    sec_id, context_chunks, chart_output_dir, ticker,
                    planned_charts=planned_charts if planned_charts else None
                )
                for cp in sec_charts:
                    fname = os.path.basename(cp)
                    report_parts.append(f"\n![{fname}]({fname})\n")
                    section_chart_paths.append(cp)
                    print(f"  [Charts] Embedded {fname} after {sec_id}")
            except Exception as exc:
                print(f"  [Charts] Failed for {sec_id}: {exc}")

    # ── Cross-section coherence check ─────────────────────────────────────────
    # One final LLM pass reviewing the concatenated report for contradictions,
    # hallucinated numbers, and internal inconsistencies. The check either
    # appends a clean bill or a list of flagged issues.
    full_draft = "\n".join(report_parts)
    print("[ReportGen] Running cross-section coherence check...")
    try:
        check_llm = get_llm(temperature=0.0, max_tokens=1024)
        check_messages = [
            SystemMessage(content=(
                "You are an internal quality checker for a financial research report. "
                "Read the full report draft below and check for:\n"
                "1. CONTRADICTIONS: Two sections stating different numbers for the same metric.\n"
                "2. UNSOURCED CLAIMS: Sections that cite specific numbers (WACC, discount rate, "
                "sensitivity parameters, price targets) that don't appear in any [FinData], "
                "[SEC XBRL], [WebSearch], [FRED], or [Calculator] tagged source block.\n"
                "3. INTERNAL INCONSISTENCY: A section claiming data is unavailable while another "
                "section uses that exact data.\n\n"
                "If you find issues, list them concisely (max 5). "
                "If the report is clean, write 'No issues found.' "
                "Do NOT rewrite the report — just flag problems."
            )),
            HumanMessage(content=f"FULL REPORT DRAFT:\n{full_draft[:12000]}"),
        ]
        check_response = invoke_with_retry(check_llm, check_messages)
        check_text = check_response.content.strip()

        if check_text and "no issues found" not in check_text.lower():
            report_parts.append(
                "\n---\n\n## Quality Review Notes\n\n"
                "*The following issues were flagged by an automated coherence check. "
                "They may indicate data gaps, contradictions, or unsourced claims:*\n\n"
                f"{check_text}"
            )
            print(f"[ReportGen] Coherence check flagged issues.")
        else:
            print(f"[ReportGen] Coherence check: clean.")
    except Exception as exc:
        print(f"[ReportGen] Coherence check failed ({exc}) — skipping.")

    final_report = "\n".join(report_parts)
    print(f"\n[ReportGen] Report complete. ~{len(final_report):,} chars.")
    return {"final_report": final_report}
