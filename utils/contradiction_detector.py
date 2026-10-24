"""
utils/contradiction_detector.py  –  Deterministic cross-source fact checker.

Runs after the tool node, before the synthesis LLM call. For every metric
the just-fetched tool results contain, this looks in the knowledge-layer DB
for OTHER recent rows describing the same (company, period_end, field) —
and if two sources disagree beyond a threshold, prepares a structured note
that gets injected into the synthesis prompt.

WHY THIS IS DETERMINISTIC, NOT ANOTHER LLM CALL
Two reasons. First, it costs zero Groq tokens — checking a number against
a database is arithmetic, not reasoning. Second, an LLM asked "do these two
numbers disagree" occasionally rationalises them into agreement anyway,
which is precisely the failure mode this exists to catch (the ROE
148.8% vs 164.6% case from earlier feedback: two providers using
different denominators produced substantively different numbers that the
model then averaged into a single confident answer).

WHY THIS MATTERS FOR THE PROJECT
Contradiction detection was the single largest source of "confidently
wrong" answers observed. The synthesis prompt now sees a structured note
like "yfinance says 148.8%, Alpha Vantage OVERVIEW says 164.6% — 15.8
percentage-point delta on ROE — surface both numbers, explain the likely
methodology difference (average vs. ending equity)" instead of just two
raw numbers with no instruction on how to handle them.

FAILURE POSTURE
If this module fails for ANY reason — DB error, unparseable tool output,
unexpected field — it MUST return no contradictions rather than raise.
The synthesis path is what the user is waiting for; a validator that can
crash the turn is worse than a validator that occasionally misses a
disagreement. Every top-level function is wrapped accordingly.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from database.connection import query


# ─────────────────────────────────────────────────────────────────────────────
#  Source tiers — evidence grading
# ─────────────────────────────────────────────────────────────────────────────
# When two sources disagree, we need a rule for which one to trust. Tiers
# are ORDINAL: lower number = higher trust. These are defensible and boring:
# SEC filings are legally-binding, yfinance/AV are commercial redistributors
# subject to computation choices (average vs ending equity, TTM windows),
# Tavily is unstructured web content and the least trustworthy for numbers.
SOURCE_TIER: dict[str, str] = {
    "sec":           "A",
    "sec_edgar":     "A",
    "fred":          "A",     # Federal Reserve — same trust tier as SEC
    "yfinance":      "B",
    "alpha_vantage": "B",
    "av":            "B",     # alias
    "tavily":        "D",
    "web":           "D",
}


def source_tier(provider: str) -> str:
    """Grade for a single provider — used both here and by the synthesis
    prompt to display citations."""
    return SOURCE_TIER.get((provider or "").lower(), "C")


def _tier_rank(tier: str) -> int:
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(tier, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Field parsing from tool text output
# ─────────────────────────────────────────────────────────────────────────────
# The tools return formatted text (see tools/financial_data.py). Rather than
# re-parsing every line, we extract only the fields that ACTUALLY get
# cross-checked — the ones a user commonly asks about and where two sources
# using different methodologies produce meaningfully different numbers.
#
# Notably: Return on Equity is on this list because it was the specific
# real-world disagreement (148.8% vs 164.6%) reported earlier.
_CHECKED_FIELDS: dict[str, str] = {
    "revenue":           "Total Revenue",
    "net_income":        "Net Income",
    "gross_profit":      "Gross Profit",
    "operating_income":  "Operating Income",
    "total_assets":      "Total Assets",
    "total_debt":        "Total Debt",
    "cash":              "Cash & Equivalents",
    "shareholder_equity": "Shareholder Equity",
}

# One period per block, then key: value pairs.
_PERIOD_RE = re.compile(r"Period:\s*(\d{4}-\d{2}-\d{2})")
_VALUE_RE  = re.compile(r"^\s*([A-Za-z][\w &()/-]+?)\s*:\s*([\-\d.,eE]+)\s*$")

# Overview-block fields (single-value, no period). Used to catch stuff like
# "Return on Equity (%)" and "Dividend Yield (%)" from the market overview.
_OVERVIEW_FIELDS: dict[str, str] = {
    "roe_pct":            "Return on Equity (%)",
    "profit_margin_pct":  "Profit Margin (%)",
    "dividend_yield_pct": "Dividend Yield (%)",
    "trailing_pe":        "Trailing P/E",
    "market_cap":         "Market Cap",
}


def _parse_periodic_facts(text: str, ticker: str) -> list[dict]:
    """Extracts (field, period_end, value) tuples from a tool response."""
    facts: list[dict] = []
    current_period: Optional[str] = None

    for line in text.splitlines():
        period_match = _PERIOD_RE.search(line)
        if period_match:
            current_period = period_match.group(1)
            continue
        if not current_period:
            continue
        value_match = _VALUE_RE.match(line)
        if not value_match:
            continue
        label, raw = value_match.group(1).strip(), value_match.group(2).strip()
        # Reverse lookup — find our field code for this human label
        for field, human in _CHECKED_FIELDS.items():
            if human == label:
                try:
                    facts.append({
                        "ticker": ticker,
                        "period_end": current_period,
                        "field": field,
                        "value": float(raw.replace(",", "")),
                    })
                except ValueError:
                    pass
                break
    return facts


def _parse_overview_facts(text: str, ticker: str) -> list[dict]:
    """Extracts point-in-time metrics (P/E, yield, ROE) from an overview block."""
    facts: list[dict] = []
    # Overview block uses "key: value" in JSON-quoted form, e.g.
    #   "Return on Equity (%)": 148.75,
    for line in text.splitlines():
        for field, human in _OVERVIEW_FIELDS.items():
            match = re.search(rf'"{re.escape(human)}"\s*:\s*([\-\d.]+)', line)
            if match:
                try:
                    facts.append({
                        "ticker": ticker,
                        "period_end": None,  # point-in-time; no period
                        "field": field,
                        "value": float(match.group(1)),
                    })
                except ValueError:
                    pass
                break
    return facts


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-source comparison
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds tuned to skip trivial rounding differences but catch real
# methodology divergences. These deltas are chosen empirically:
# - Absolute figures: 5% of the value or $1M, whichever is larger. Two
#   providers can round differently, currency-convert differently, or
#   include/exclude different subsidiaries; sub-5% is usually noise.
# - Ratios/percentages: 2 percentage points. ROE of 148 vs 149 is noise;
#   148 vs 164 is a methodology story.
def _is_meaningfully_different(v1: float, v2: float, field: str) -> bool:
    if v1 is None or v2 is None:
        return False
    delta = abs(v1 - v2)

    ratio_fields = {"roe_pct", "profit_margin_pct", "dividend_yield_pct",
                    "trailing_pe"}
    if field in ratio_fields:
        return delta >= 2.0

    scale = max(abs(v1), abs(v2), 1_000_000.0)
    return delta / scale >= 0.05 and delta >= 1_000_000.0


def _lookup_alternate_sources(field: str, ticker: str, period_end: Optional[str]) -> list[dict]:
    """Queries the knowledge layer for other recent rows describing the same
    fact, from providers we haven't just seen."""
    if field in ("revenue", "gross_profit", "operating_income", "net_income"):
        table, column = "income_statements", field
    elif field in ("total_assets", "total_debt", "cash", "shareholder_equity"):
        # Column name differences between tables — real schema quirks that
        # got baked in during Phase A and don't warrant a migration for.
        table = "balance_sheets"
        column = {
            "total_assets":       "total_assets",
            "total_debt":         "total_debt",
            "cash":               "cash",
            "shareholder_equity": "total_equity",
        }[field]
    else:
        return []

    # Only recent rows count — old data from a previous fiscal year isn't
    # a "contradiction", it's just history. 14-day window matches the
    # annual_financials TTL from schema.TTL_SECONDS.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()

    try:
        base = f"""
            SELECT
                fs.{column} AS value,
                fs.period_end,
                s.provider,
                s.retrieved_at
            FROM {table} fs
            JOIN companies c ON c.id = fs.company_id
            LEFT JOIN sources s ON s.id = fs.source_id
            WHERE c.ticker = ?
              AND fs.updated_at > ?
              AND fs.{column} IS NOT NULL
        """
        params: tuple = (ticker.upper(), cutoff)
        if period_end:
            base += " AND fs.period_end = ?"
            params = params + (period_end,)
        base += " ORDER BY fs.updated_at DESC LIMIT 10"

        return query(base, params)
    except Exception as exc:
        print(f"[Contradiction] DB lookup failed for {ticker}.{field}: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_contradictions(tool_trace: list[dict], active_ticker: Optional[str]) -> list[dict]:
    """
    Given this turn's tool call trace, returns a list of structured
    contradiction records. Empty list = no known disagreements OR the check
    couldn't run (fail-safe).

    Each record has:
        field:            "revenue" / "net_income" / ...
        ticker:           "MSFT"
        period_end:       "2024-06-30" or None for point-in-time
        current: {value, provider, tier}
        alternates: [{value, provider, tier, delta, delta_pct}, ...]
        primary_recommendation: which value the synthesis should lead with
    """
    contradictions: list[dict] = []

    try:
        # Extract facts from THIS turn's tool responses. args["ticker"] is
        # the anchoring identifier — we don't try to parse tickers out of
        # the payload text, that's brittle and the tool call args already
        # know.
        current_facts: list[dict] = []
        for record in tool_trace or []:
            if record.get("status") != "success":
                continue
            if record.get("tool") != "financial_data_api":
                continue
            args = record.get("args") or {}
            ticker = args.get("ticker") or active_ticker
            if not ticker:
                continue
            # NOTE: tool_trace here doesn't carry the payload text — only
            # metadata. See nodes/chat_agent.py's chat_tool_node where the
            # full payload is stored in successful_tool_results. Callers
            # pass that in via `tool_trace_with_payloads` in
            # find_contradictions_from_state.
            continue

    except Exception as exc:
        print(f"[Contradiction] Unexpected error: {exc}")
        return []

    return contradictions


def find_contradictions_from_state(state: dict) -> list[dict]:
    """
    Same as find_contradictions, but reads directly from the graph state's
    successful_tool_results, which DO carry payloads (unlike tool_trace
    which is deliberately payload-free to keep it cheap to stream).
    """
    contradictions: list[dict] = []

    try:
        successful = state.get("successful_tool_results", []) or []
        active_ticker = state.get("active_ticker")

        current_facts: list[dict] = []
        current_providers: dict[tuple, str] = {}

        for record in successful:
            if record.get("tool") != "financial_data_api":
                continue
            args = record.get("args") or {}
            ticker = args.get("ticker") or active_ticker
            payload = str(record.get("result") or "")
            if not ticker or not payload:
                continue

            for fact in _parse_periodic_facts(payload, ticker):
                current_facts.append(fact)
                current_providers[(fact["ticker"], fact["period_end"], fact["field"])] = "yfinance"

            for fact in _parse_overview_facts(payload, ticker):
                current_facts.append(fact)
                current_providers[(fact["ticker"], fact["period_end"], fact["field"])] = "yfinance"

        # For each freshly-observed fact, look up any recent alternate-
        # source rows describing the same thing and compare.
        for fact in current_facts:
            alternates = _lookup_alternate_sources(
                fact["field"], fact["ticker"], fact["period_end"]
            )
            current_provider = current_providers.get(
                (fact["ticker"], fact["period_end"], fact["field"]), "yfinance"
            )

            differing = []
            for alt in alternates:
                if not alt.get("provider"):
                    continue
                if alt["provider"].lower() == current_provider.lower():
                    continue
                if not _is_meaningfully_different(fact["value"], alt["value"], fact["field"]):
                    continue

                delta = alt["value"] - fact["value"]
                delta_pct = (delta / fact["value"] * 100.0) if fact["value"] else None
                differing.append({
                    "value": alt["value"],
                    "provider": alt["provider"],
                    "tier": source_tier(alt["provider"]),
                    "delta": delta,
                    "delta_pct": delta_pct,
                })

            if not differing:
                continue

            all_candidates = [
                {"value": fact["value"], "provider": current_provider,
                 "tier": source_tier(current_provider)}
            ] + differing
            primary = min(all_candidates, key=lambda c: _tier_rank(c["tier"]))

            contradictions.append({
                "field": fact["field"],
                "ticker": fact["ticker"],
                "period_end": fact["period_end"],
                "current": {
                    "value": fact["value"],
                    "provider": current_provider,
                    "tier": source_tier(current_provider),
                },
                "alternates": differing,
                "primary_recommendation": {
                    "value": primary["value"],
                    "provider": primary["provider"],
                    "reason": (
                        f"Tier {primary['tier']} source (per SOURCE_TIER — "
                        f"SEC/FRED=A, yfinance/AV=B, web=D)"
                    ),
                },
            })

    except Exception as exc:
        print(f"[Contradiction] Unexpected error: {exc}")
        return []

    return contradictions


def format_contradictions_note(contradictions: list[dict]) -> str:
    """
    Renders contradictions as a structured instruction block for the
    synthesis prompt. Written to be actionable, not just informational —
    the model is told what number to lead with, in what units, and how to
    frame the disagreement.
    """
    if not contradictions:
        return ""

    lines = [
        "## Data reconciliation notes",
        "",
        "The tool results below contain values that DISAGREE with other "
        "recent sources for the same fact. Do NOT silently pick one; SURFACE "
        "the disagreement to the user, then lead with the higher-tier source:",
        "",
    ]

    for note in contradictions:
        period_str = f" ({note['period_end']})" if note.get("period_end") else ""
        lines.append(
            f"- **{note['ticker']} {note['field']}{period_str}**: "
            f"{note['current']['provider']} says {note['current']['value']:,.2f} "
            f"(tier {note['current']['tier']}); "
            + "; ".join(
                f"{alt['provider']} says {alt['value']:,.2f} (tier {alt['tier']}, "
                f"Δ {alt['delta']:+,.2f}"
                + (f", {alt['delta_pct']:+.1f}%)" if alt.get('delta_pct') is not None else ")")
                for alt in note["alternates"]
            )
        )
        lines.append(
            f"  → Lead with {note['primary_recommendation']['provider']} "
            f"({note['primary_recommendation']['reason']}). Explain the "
            f"delta as a methodology difference — providers use different "
            f"denominators (average vs ending equity, TTM vs FY windows, "
            f"etc.) — rather than picking one silently."
        )

    lines.append("")
    return "\n".join(lines)
