"""
tools/__init__.py  –  Central registry for all agent tools.

Exports:
    ALL_TOOLS        – list of LangChain @tool decorated functions
    TOOL_NODE        – LangGraph ToolNode ready to drop into the graph
    TOOL_DESCRIPTIONS – plain-text summary for roadmap planner prompt
"""

import re
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import ToolNode
from typing import Optional, Union

from .sec_filing           import sec_filing_search   as _sec
from .web_search           import web_search          as _web
from .earnings_transcript  import earnings_transcript as _transcript
from .financial_data       import financial_data_api  as _findata
from .calculator           import calculator          as _calc
from .fred                 import fred_series          as _fred


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_quarter(quarter: Union[str, int]) -> str:
    """
    Coerces any quarter representation to 'Q1'–'Q4' string.
    Handles: 1, 2, 3, 4, "1", "2", "Q1", "q2", "Q4" etc.
    """
    q = str(quarter).upper().strip()
    if not q.startswith("Q"):
        q = f"Q{q}"
    return q   # e.g. "Q4"


# ── Plain callables ───────────────────────────────────────────────────────────
# These are undecorated, ordinary Python functions. The deterministic fetch
# executor calls these directly (no LLM involved) using args taken straight
# from the roadmap plan. The @lc_tool-wrapped versions below exist only for
# the compute executor's forced-tool_choice LLM calls (currently just
# calculator) — fetch tools no longer need an LLM-facing schema at all,
# but we keep thin wrappers for them too so ALL_TOOLS / TOOL_NODE still work
# for anything that inspects the tool list generically.

# ── Caching ──────────────────────────────────────────────────────────────────
# Every fetch tool routes through cache_service here rather than each tool
# implementing its own caching. This is the single chokepoint that BOTH the
# chat agent and the report pipeline already funnel through, so one wiring
# point covers the whole project — and the TTL policy stays in one file
# (database/schema.py) instead of being scattered across six tools.
#
# The calculator is deliberately NOT cached: it's pure local arithmetic with
# no network call and no rate limit, so caching it would add DB round-trips
# to make free operations slower.
from database.cache_service import cached_call


def sec_filing_search_fn(
    ticker: str,
    filing_type: str = "10-K",
    year: Optional[int] = None,
) -> str:
    args = {"ticker": ticker, "filing_type": filing_type, "year": year}

    def _normalize(payload: str, source_id):
        from database.normalizer import normalize_sec_xbrl_text
        period_type = "quarterly" if filing_type == "10-Q" else "annual"
        normalize_sec_xbrl_text(ticker, payload, source_id, period_type=period_type)

    payload, _ = cached_call(
        "sec_filing_search", args, "sec_filing",
        lambda: _sec(ticker=ticker, filing_type=filing_type, year=year),
        normalize=_normalize,
        provider="sec",
        source_type="filing",
    )
    return payload


def web_search_fn(
    query: str,
    num_results: int = 10,
    date_range: Optional[str] = None,
) -> str:
    args = {"query": query, "num_results": num_results, "date_range": date_range}
    # Provenance ONLY — no structured normalizer. News/search results are
    # unstructured prose with no numeric fact to extract into typed columns,
    # and contradiction detection only operates on numeric facts anyway, so
    # a "news_articles" table with parsed rows wouldn't earn its complexity.
    # What DOES matter and is genuinely cheap: recording that a Tavily call
    # happened, when, for what query — real provenance for "as of when did
    # we search for this" without pretending prose can be normalized like a
    # balance sheet.
    payload, _ = cached_call(
        "web_search", args, "news",
        lambda: _web(query=query, num_results=num_results, date_range=date_range),
        provider="tavily",
        source_type="web",
    )
    return payload


def earnings_transcript_fn(
    ticker: str,
    quarter: Union[str, int],
    year: int,
) -> str:
    normalised_quarter = _normalise_quarter(quarter)
    args = {"ticker": ticker, "quarter": normalised_quarter, "year": year}
    # The highest-value cache in the project: Alpha Vantage allows 25 calls
    # per DAY on the free tier, and a published transcript's text never
    # changes once released, so re-fetching one is pure waste of a scarce
    # resource.
    payload, _ = cached_call(
        "earnings_transcript", args, "earnings_transcript",
        lambda: _transcript(ticker=ticker, quarter=normalised_quarter, year=year),
        provider="alpha_vantage",
        source_type="transcript",
    )
    return payload


def financial_data_api_fn(
    ticker: str,
    statement_type: str = "all",
    period: str = "annual",
    years: int = 3,
    fields: Optional[list[str]] = None,
) -> str:
    """
    fields is a NEW response-shaping parameter — see _filter_fields below for
    what it accepts. Kept optional (defaults to full response) so nothing
    that already works gets broken; documented in the LangChain tool wrapper
    so the model can request selective returns for comparison queries.

    Deliberately applied AFTER the DB rebuild / cache read / live fetch, not
    before. This is response-SHAPING not scope-scoping: the DB and cache
    still hold the full response so the next request that wants a different
    slice can be answered from cache. Filtering earlier would trade a smaller
    LLM prompt for many more misses in the DB/cache layer, which is the
    opposite of the trade we want.
    """
    args = {
        "ticker": ticker, "statement_type": statement_type,
        "period": period, "years": years,
        # fields deliberately NOT in cache key — a call with fields=["revenue"]
        # must hit the same cache entry as one without fields, because the
        # underlying fetch is identical and only the return shape differs.
    }

    # DB-first: if the knowledge layer has fresh typed rows for this
    # (ticker, statement, period_type, years), rebuild the text response
    # from those rows instead of calling the API OR touching the text
    # cache. Runs before cached_call because cache_key includes years and
    # statement_type — so a text-cache hit needs identical args, but the
    # DB can serve DIFFERENT arg combos from the same underlying rows.
    # That's the main API-call reduction win: chat's "revenue over 2
    # years" and a later report's "income statement, 3 years" both come
    # from the DB even though they'd MISS in the text cache.
    try:
        from database.financial_reads import try_rebuild_financials
        rebuilt = try_rebuild_financials(ticker, statement_type, period, years)
        if rebuilt:
            print(f"[FinData] Served from DB (no API call): {ticker} {statement_type} {period}")
            return _filter_fields(rebuilt, fields)
    except Exception as exc:
        # Same failure posture as caching: DB-layer error becomes a MISS,
        # never a crash. Falls through to cached_call below exactly as if
        # this branch didn't exist.
        print(f"[FinData] DB read failed ({exc}) — falling through to cache/API.")

    ttl_class = "annual_financials" if period == "annual" else "quarterly_financials"

    def _normalize(payload: str, source_id):
        from database.normalizer import normalize_yfinance_text
        normalize_yfinance_text(ticker, payload, source_id, period_type=period)

    payload, _ = cached_call(
        "financial_data_api", args, ttl_class,
        lambda: _findata(
            ticker=ticker, statement_type=statement_type, period=period, years=years
        ),
        normalize=_normalize,
        provider="yfinance",
    )
    return _filter_fields(payload, fields)


# ── Field filtering ─────────────────────────────────────────────────────────
# Aliases the model can pass in `fields`. Kept small on purpose — the whole
# point is to be a light hint layer, not a query language. If a request needs
# a field not in this map, it falls through to returning the full text
# unchanged, which is strictly safer than dropping data the model needed.
_FIELD_ALIASES: dict[str, list[str]] = {
    "revenue":          ["Total Revenue", "Revenue (TTM)"],
    "net_income":       ["Net Income", "Net Income (TTM)"],
    "gross_profit":     ["Gross Profit"],
    "gross_margin":     ["Gross Profit", "Total Revenue"],
    "operating_income": ["Operating Income"],
    "operating_margin": ["Operating Income", "Total Revenue"],
    "net_margin":       ["Profit Margin (%)", "Net Income", "Total Revenue"],
    "eps":              ["EPS (Basic)", "EPS (Diluted)"],
    "pe":               ["Trailing P/E", "Forward P/E"],
    "market_cap":       ["Market Cap"],
    "cash":             ["Cash & Equivalents"],
    "debt":             ["Total Debt", "Debt/Equity"],
    "assets":           ["Total Assets"],
    "equity":           ["Shareholder Equity", "Return on Equity (%)"],
    "roe":              ["Return on Equity (%)"],
    "cash_flow":        ["Operating Cash Flow", "Free Cash Flow", "CapEx"],
    "free_cash_flow":   ["Free Cash Flow", "Operating Cash Flow", "CapEx"],
    "price":            ["Current Price", "52-Week High", "52-Week Low"],
    "dividend":         ["Dividend Yield (%)"],
    "beta":             ["Beta"],
    "summary":          ["Business Summary", "Sector", "Industry"],
}


def _filter_fields(payload: str, fields: Optional[list[str]]) -> str:
    """Trims a full financial_data response to only the requested field lines,
    keeping section headers so the output still parses like a normal response."""
    if not fields or not payload:
        return payload

    wanted_labels: set[str] = set()
    for field in fields:
        aliases = _FIELD_ALIASES.get(field.lower().replace(" ", "_"), [])
        wanted_labels.update(aliases)

    if not wanted_labels:
        # No known aliases matched — safer to return everything than to
        # silently drop what the caller asked for.
        return payload

    kept_lines: list[str] = []
    for line in payload.splitlines():
        stripped = line.strip()
        # Keep structural lines (headers, period markers, blank separators)
        # so the output still looks like a normal response block.
        if (not stripped
            or stripped.startswith("[FinData]")
            or stripped.startswith("=")
            or stripped.startswith("Period:")
            or stripped.startswith("{")
            or stripped.startswith("}")):
            kept_lines.append(line)
            continue

        # Match either "  Total Revenue: 123" (statement format) or
        # `"Total Revenue": 123` (overview JSON format). Statement lines
        # can have padded whitespace before the colon ("Total Revenue      :
        # 123") — the substring check normalizes that away by removing
        # the whitespace-before-colon before matching.
        normalized = re.sub(r"\s+:", ":", stripped)
        if any(
            f"{label}:" in normalized or f'"{label}"' in stripped
            for label in wanted_labels
        ):
            kept_lines.append(line)

    result = "\n".join(kept_lines)
    # If filtering left nothing useful (label mismatch), fall back to the
    # full payload rather than returning an empty block.
    if len(result) < len(payload) * 0.15:
        return result if result.strip() else payload
    return result


def fred_series_fn(
    series_id: str,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    limit: int = 60,
) -> str:
    args = {
        "series_id": series_id, "observation_start": observation_start,
        "observation_end": observation_end, "limit": limit,
    }
    def _normalize(payload: str, source_id):
        from database.normalizer import normalize_fred_series
        normalize_fred_series(series_id, series_id, payload, source_id)

    payload, _ = cached_call(
        "fred_series", args, "macro_series",
        lambda: _fred(
            series_id=series_id,
            observation_start=observation_start,
            observation_end=observation_end,
            limit=limit,
        ),
        normalize=_normalize,
        provider="fred",
    )
    return payload


# Direct name -> plain callable, used by the deterministic fetch executor.
FETCH_TOOL_FUNCTIONS = {
    "sec_filing_search":  sec_filing_search_fn,
    "web_search":         web_search_fn,
    "earnings_transcript": earnings_transcript_fn,
    "financial_data_api": financial_data_api_fn,
    "fred_series":        fred_series_fn,
}


# ── LLM-facing wrappers ───────────────────────────────────────────────────────
# Each function is wrapped with @lc_tool so LangChain generates the JSON schema
# the LLM reads when deciding which tool to call / filling in arguments.

@lc_tool
def sec_filing_search(
    ticker: str,
    filing_type: str = "10-K",
    year: Optional[int] = None,
) -> str:
    """Retrieves SEC filings (10-K, 10-Q, 8-K, DEF 14A) from EDGAR for a given
    company ticker and filing type. Returns filing text, date, and accession number.
    Args:
        ticker: Stock ticker e.g. AAPL
        filing_type: One of 10-K, 10-Q, 8-K, DEF 14A
        year: Optional calendar year e.g. 2024
    """
    return sec_filing_search_fn(ticker=ticker, filing_type=filing_type, year=year)


@lc_tool
def web_search(
    query: str,
    num_results: int = 10,
    date_range: Optional[str] = None,
) -> str:
    """Performs a web search for current news, analysis, and commentary about a
    company or financial topic. Returns list of URL, title, and content snippets.
    Args:
        query: Search string e.g. 'AAPL Q4 2024 earnings analysis'
        num_results: Number of results to return (default 10)
        date_range: Optional number of days back as string e.g. '7'
    """
    return web_search_fn(query=query, num_results=num_results, date_range=date_range)


@lc_tool
def earnings_transcript(
    ticker: str,
    quarter: Union[str, int],
    year: int,
) -> str:
    """Retrieves earnings call transcript for a specific company, quarter, and year.
    Returns full transcript text with speaker labels and sentiment tags.
    Args:
        ticker: Stock ticker e.g. AAPL or TCS.NS
        quarter: Quarter as string OR int. Accepts: 'Q1','Q2','Q3','Q4' OR 1,2,3,4
        year: Calendar year as integer e.g. 2024
    """
    return earnings_transcript_fn(ticker=ticker, quarter=quarter, year=year)


@lc_tool
def financial_data_api(
    ticker: str,
    statement_type: str = "all",
    period: str = "annual",
    years: int = 3,
    fields: Optional[list[str]] = None,
) -> str:
    """Retrieves structured financial data: income statement, balance sheet, cash flow,
    key ratios, and market overview.
    Args:
        ticker: Stock ticker e.g. AAPL or TCS.NS
        statement_type: One of overview, income, balance, cashflow, ratios, all
        period: 'annual' or 'quarterly'
        years: Number of periods to retrieve (1-5)
        fields: OPTIONAL list of specific fields you want returned. Cuts response
            size dramatically — use this for comparison queries where you only
            need one or two metrics per company. Recognized aliases: revenue,
            net_income, gross_profit, gross_margin, operating_income,
            operating_margin, net_margin, eps, pe, market_cap, cash, debt, assets,
            equity, roe, cash_flow, free_cash_flow, price, dividend, beta.
            Example: fields=["revenue", "operating_margin"] when comparing
            profitability across companies. Omit to get the full response.
    """
    return financial_data_api_fn(
        ticker=ticker, statement_type=statement_type,
        period=period, years=years, fields=fields,
    )


@lc_tool
def calculator(
    metric: str,
    start_value: Optional[float] = None,
    end_value: Optional[float] = None,
    years: Optional[float] = None,
    operating_cash_flow: Optional[float] = None,
    capex: Optional[float] = None,
    fcf: Optional[float] = None,
    market_cap: Optional[float] = None,
    working_capital: Optional[float] = None,
    retained_earnings: Optional[float] = None,
    ebit: Optional[float] = None,
    total_liabilities: Optional[float] = None,
    revenue: Optional[float] = None,
    total_assets: Optional[float] = None,
    total_debt: Optional[float] = None,
    shareholder_equity: Optional[float] = None,
    current_assets: Optional[float] = None,
    current_liabilities: Optional[float] = None,
    cogs: Optional[float] = None,
    gross_profit: Optional[float] = None,
    current_revenue: Optional[float] = None,
    prior_revenue: Optional[float] = None,
    pe_ratio: Optional[float] = None,
    earnings_growth_rate: Optional[float] = None,
    cash: Optional[float] = None,
    net_income: Optional[float] = None,
    interest_expense: Optional[float] = None,
) -> str:
    """Calculate one financial metric. Pass only the fields the chosen metric needs.

    metric: cagr | free_cash_flow | fcf_yield | altman_z_score | debt_to_equity
      | current_ratio | gross_margin | revenue_growth | peg_ratio
      | enterprise_value | roe | interest_coverage

    Fields per metric:
      cagr: start_value, end_value, years
      free_cash_flow: operating_cash_flow, capex
      fcf_yield: fcf, market_cap
      altman_z_score: working_capital, retained_earnings, ebit, market_cap,
        total_liabilities, revenue, total_assets
      debt_to_equity: total_debt, shareholder_equity
      current_ratio: current_assets, current_liabilities
      gross_margin: revenue + (cogs OR gross_profit; use whichever you have,
        most APIs return gross_profit directly)
      revenue_growth: current_revenue, prior_revenue
      peg_ratio: pe_ratio, earnings_growth_rate
      enterprise_value: market_cap, total_debt, cash
      roe: net_income, shareholder_equity
      interest_coverage: ebit, interest_expense
    """
    return _calc(
        metric=metric,
        start_value=start_value, end_value=end_value, years=years,
        operating_cash_flow=operating_cash_flow, capex=capex, fcf=fcf,
        market_cap=market_cap, working_capital=working_capital,
        retained_earnings=retained_earnings, ebit=ebit,
        total_liabilities=total_liabilities, revenue=revenue,
        total_assets=total_assets, total_debt=total_debt,
        shareholder_equity=shareholder_equity, current_assets=current_assets,
        current_liabilities=current_liabilities, cogs=cogs,
        gross_profit=gross_profit,
        current_revenue=current_revenue, prior_revenue=prior_revenue,
        pe_ratio=pe_ratio, earnings_growth_rate=earnings_growth_rate,
        cash=cash, net_income=net_income, interest_expense=interest_expense,
    )


# ── Export list and ToolNode ──────────────────────────────────────────────────

@lc_tool
def fred_series(
    series_id: str,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    limit: int = 60,
) -> str:
    """Fetches a macroeconomic time-series from FRED (Federal Reserve Economic Data).
    Args:
        series_id: Either a shortcut (gdp, cpi, core_cpi, fed_funds, unemployment,
                    10y_yield, 2y_yield, yield_curve, industrial_prod, m2,
                    pce_inflation, retail_sales, housing_starts, consumer_sent)
                    or a raw FRED series ID (e.g. "GDP", "CPIAUCSL", "DGS10").
        observation_start: Optional ISO date "YYYY-MM-DD" — earliest date to fetch.
        observation_end: Optional ISO date "YYYY-MM-DD" — latest date to fetch.
        limit: Max observations to return (default 60).
    """
    return fred_series_fn(
        series_id=series_id,
        observation_start=observation_start,
        observation_end=observation_end,
        limit=limit,
    )


ALL_TOOLS = [
    sec_filing_search,
    web_search,
    earnings_transcript,
    financial_data_api,
    calculator,
    fred_series,
]

# Name -> LLM-facing tool, used by the compute executor to force tool_choice
# onto a single named tool.
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

TOOL_NODE = ToolNode(ALL_TOOLS)

# Plain-text tool menu for embedding in LLM prompts
TOOL_DESCRIPTIONS = """
Available tools:
1. sec_filing_search(ticker, filing_type, year)
   → Fetches 10-K / 10-Q / 8-K / DEF 14A text + XBRL numbers from EDGAR.

2. web_search(query, num_results, date_range)
   → Tavily web search for news, analyst commentary, recent events.

3. earnings_transcript(ticker, quarter, year)
   → Full earnings call transcript with speaker labels (Alpha Vantage).
   → quarter accepts BOTH string ('Q4') and integer (4) — either works.

4. financial_data_api(ticker, statement_type, period, years)
   → Income statement, balance sheet, cash flow, ratios (yfinance + AV).
   → statement_type options: overview | income | balance | cashflow | ratios | all

5. calculator(metric, ...named numeric fields...)
   → Derives custom indices: CAGR, FCF Yield, Altman Z-Score, ROE, PEG, etc.
   → Every input is a real named field, not a generic kwargs blob. Required
     fields per metric:
       cagr               -> start_value, end_value, years
       free_cash_flow     -> operating_cash_flow, capex
       fcf_yield          -> fcf, market_cap
       altman_z_score     -> working_capital, retained_earnings, ebit,
                             market_cap, total_liabilities, revenue, total_assets
       debt_to_equity     -> total_debt, shareholder_equity
       current_ratio      -> current_assets, current_liabilities
       gross_margin       -> revenue, AND EITHER cogs OR gross_profit
       revenue_growth     -> current_revenue, prior_revenue
       peg_ratio          -> pe_ratio, earnings_growth_rate
       enterprise_value   -> market_cap, total_debt, cash
       roe                -> net_income, shareholder_equity
       interest_coverage  -> ebit, interest_expense

6. fred_series(series_id, observation_start, observation_end, limit)
   → US and international macroeconomic time-series from FRED (Federal Reserve
     Economic Data). Use for MACRO_COUNTRY reports and macro context in
     earnings/event reports.
   → series_id shortcuts: gdp, cpi, core_cpi, fed_funds, unemployment,
     10y_yield, 2y_yield, yield_curve, industrial_prod, m2, pce_inflation,
     retail_sales, housing_starts, consumer_sent — or any raw FRED series ID.
"""