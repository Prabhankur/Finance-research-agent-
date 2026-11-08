"""
tools/__init__.py  –  Central registry for all agent tools.

Exports:
    ALL_TOOLS        – list of LangChain @tool decorated functions
    TOOL_NODE        – LangGraph ToolNode ready to drop into the graph
    TOOL_DESCRIPTIONS – plain-text summary for roadmap planner prompt
"""

import re
from concurrent.futures import ThreadPoolExecutor
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import ToolNode
from typing import Optional, Union

from .sec_filing           import sec_filing_search   as _sec
from .web_search           import web_search          as _web
from .earnings_transcript  import earnings_transcript as _transcript
from .financial_data       import financial_data_api  as _findata
from .calculator           import calculator          as _calc
from .fx                   import fx_rate             as _fx
from .fred                 import fred_series          as _fred, resolve_series_id


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
    # Canonical spellings from here on, so the cache key, the typed tables and the
    # live fetch all see the same thing: " msft" must not create a second company
    # row, nor a second cache entry.
    ticker = (ticker or "").strip().upper()
    statement_type = (statement_type or "all").strip().lower()
    period = (period or "annual").strip().lower()
    args = {
        "ticker": ticker, "statement_type": statement_type,
        "period": period, "years": years,
        # fields deliberately NOT in cache key — a call with fields=["revenue"]
        # must hit the same cache entry as one without fields, because the
        # underlying fetch is identical and only the return shape differs.
    }

    ttl_class = "annual_financials" if period == "annual" else "quarterly_financials"

    # LAYER 1 — exact-match text cache. Checked FIRST, on purpose:
    #   * it is one indexed lookup; the DB rebuild is up to three queries plus
    #     string reconstruction;
    #   * it holds the ORIGINAL response, byte for byte — including the Market
    #     Overview and Key Ratios, which the typed tables cannot rebuild — so a
    #     repeat of the same request gets the same answer, not a thinner copy;
    #   * it can never be worse than the layer below it: when both would hit,
    #     this one is richer; when only the DB would hit (a differently-shaped
    #     request), this one misses and costs nothing.
    # Before this, the DB rebuild ran first, so an identical repeat was answered
    # with the lossy reconstruction even though the full original was cached.
    try:
        from database.cache_service import get_cached, make_cache_key
        exact = get_cached(make_cache_key("financial_data_api", args))
        if exact:
            print(f"[Cache] HIT financial_data_api ({ttl_class}) — exact match")
            return _filter_fields(exact, fields)
    except Exception as exc:
        print(f"[FinData] cache probe failed ({exc}) — continuing to the DB layer.")

    # LAYER 2 — knowledge layer. If the typed tables hold fresh rows for this
    # (ticker, statement, period_type, years), rebuild the text response from
    # them instead of calling the API. This is what the text cache cannot do:
    # its key includes `years` and `statement_type`, so it needs identical
    # args, while the DB can serve DIFFERENT arg combos from the same rows —
    # "revenue over 2 years" now and "income statement, 3 years" later.
    # Single statements only (see try_rebuild_financials for why not "all").
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

    # LAYER 3 — live provider. cached_call re-checks the text cache (harmless),
    # then fetches, records provenance, normalizes and stores.
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
    # Resolve shortcuts ("gdp" -> "GDP") BEFORE the cache key is built, so a shortcut
    # and the real series code share one entry.
    series_id = resolve_series_id(series_id)
    args = {
        "series_id": series_id, "observation_start": observation_start,
        "observation_end": observation_end, "limit": limit,
    }
    def _normalize(payload: str, source_id):
        from database.normalizer import normalize_fred_series
        from tools.fred import resolve_series_id
        # Key by the REAL series id: the model may pass the shortcut "gdp",
        # and registering that as its own series would duplicate "GDP".
        code = resolve_series_id(series_id)
        normalize_fred_series(code, code, payload, source_id)

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


def fx_rate_fn(from_currency: str, to_currency: str) -> str:
    frm, to = (from_currency or "").strip().upper(), (to_currency or "").strip().upper()
    args = {"from_currency": frm, "to_currency": to}
    # "valuation" TTL (1 hour): a market rate drifts through the day, and one
    # cached hour is plenty for "convert this figure to rupees". Error payloads
    # are never cached (see cache_service._looks_like_error).
    payload, _ = cached_call(
        "fx_rate", args, "valuation",
        lambda: _fx(from_currency=frm, to_currency=to),
        provider="yfinance", source_type="api",
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


# ── multi-company calls ───────────────────────────────────────────────────────
# A question about several companies used to need one financial_data_api call per
# company, and it was up to the MODEL to put them all in one message. In a real
# run it did not: it made one call per round, hit the 4-round budget after four
# companies, and the other four came back "not fetched". The fix is in the tool's
# shape rather than a plea in the prompt: `tickers` takes the whole list, so ONE
# call carries every company and the code fans it out in parallel.
from config import CHAT_MAX_BATCH_TICKERS as MAX_BATCH_TICKERS

# Up to 20 characters: Indian tickers with an exchange suffix are long ("BAJFINANCE.NS" is 13).
_TICKER_TOKEN = re.compile(r"^\^?[A-Z][A-Z0-9.\-=]{0,19}$")


def normalize_tickers(ticker=None, tickers=None, limit: int = MAX_BATCH_TICKERS) -> tuple[list[str], list[str]]:
    """
    (kept, skipped) from a single ticker and/or a list. Upper-cases, removes
    duplicates keeping first-seen order, and caps at `limit`.

    `skipped` is everything NOT fetched — overflow past the cap AND anything that
    is not shaped like a ticker (a company NAME such as "Apple Inc", say) — so the
    caller can say so. Nothing is dropped silently: a missing company the model
    does not know about is how a comparison ends up quietly short.

    Splitting: a bare string (the `ticker` argument, or `tickers` given as one
    string) is split on commas, semicolons and whitespace, because models do pass
    "AAPL MSFT". An item INSIDE a list is split on commas only, so "Apple Inc"
    stays one (invalid) token instead of becoming the fake tickers APPLE and INC.
    """
    raw: list[str] = []
    for source in (ticker, tickers):
        if source is None:
            continue
        if isinstance(source, str):
            raw.extend(re.split(r"[,;\s]+", source))
        elif isinstance(source, (list, tuple)):
            for item in source:
                if isinstance(item, str):
                    raw.extend(re.split(r"[,;]+", item))
    seen: set[str] = set()
    ordered: list[str] = []
    invalid: list[str] = []
    for token in raw:
        token = token.strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        (ordered if _TICKER_TOKEN.match(token) else invalid).append(token)
    return ordered[:limit], ordered[limit:] + invalid


def merge_batch_results(entries: list[dict], skipped: Optional[list[str]] = None) -> str:
    """
    One text block for a multi-company call. Each company sits under an explicit
    `=== TICKER ===` header so the model (and a human reading a partial result)
    can tell them apart, and a one-line summary says which ones failed, so a
    partial batch is visible instead of looking complete.
    entries: [{"ticker", "status": "success" | "failed" | "empty", "content"}]
    """
    ok = [e for e in entries if e["status"] == "success"]
    bad = [e["ticker"] for e in entries if e["status"] != "success"]
    head = (f"[Batch] financial_data_api for {len(entries)} companies "
            f"(fetched in parallel): {len(ok)} ok")
    if bad:
        head += f", {len(bad)} failed or empty ({', '.join(bad)})"
    parts = [head]
    for e in entries:
        parts.append(f"=== {e['ticker']} ===\n{e['content']}")
    if skipped:
        parts.append(f"[Batch] Not fetched (over the {MAX_BATCH_TICKERS}-per-call limit, or not a "
                     f"valid ticker symbol): {', '.join(skipped)} — ask again using tickers.")
    return "\n\n".join(parts)


def financial_data_api_batch_fn(tickers, statement_type: str = "all", period: str = "annual",
                                years: int = 3, fields=None) -> str:
    """
    Standalone fan-out (used when the tool is invoked directly). In chat the tool
    node expands the call itself — so each company goes through the same
    retry / withdraw / trace policy as any other tool call — and then uses
    merge_batch_results for the identical output format.
    """
    from utils.error_classifier import OK, LEGITIMATE_EMPTY, classify_result
    kept, skipped = normalize_tickers(tickers=tickers)
    if not kept:
        return "[financial_data_api] Error: provide `ticker`, or a non-empty `tickers` list."

    def one(t: str) -> dict:
        try:
            content = financial_data_api_fn(ticker=t, statement_type=statement_type,
                                            period=period, years=years, fields=fields)
        except Exception as exc:
            content = f"[financial_data_api] Error: raised {type(exc).__name__}: {exc}"
        category = classify_result("financial_data_api", content)
        status = "success" if category == OK else "empty" if category == LEGITIMATE_EMPTY else "failed"
        return {"ticker": t, "status": status, "content": content}

    with ThreadPoolExecutor(max_workers=len(kept)) as pool:
        entries = list(pool.map(one, kept))
    return merge_batch_results(entries, skipped)


@lc_tool
def financial_data_api(
    ticker: str = "",
    tickers: Optional[list[str]] = None,
    statement_type: str = "all",
    period: str = "annual",
    years: int = 3,
    fields: Optional[list[str]] = None,
) -> str:
    """Retrieves structured financial data: income statement, balance sheet, cash flow,
    key ratios, and market overview.
    Args:
        ticker: ONE company's ticker. US listings are bare (AAPL). Other markets NEED the
            exchange suffix: .NS NSE India, .BO BSE India, .L London, .TO Toronto
            (e.g. RELIANCE.NS, SBIN.NS). A bare ticker always means the US listing.
        tickers: TWO OR MORE companies: pass ALL of them here in ONE call (max 8), e.g.
            ["AAPL","MSFT","NVDA"]. They are fetched in parallel. Never make one call per company.
        statement_type: One of overview, income, balance, cashflow, ratios, all
        period: 'annual' or 'quarterly'
        years: Number of periods to retrieve (1-5)
        fields: OPTIONAL list of specific fields you want returned. Cuts response
            size dramatically — use this for comparison queries where you only
            need one or two metrics per company. Recognized aliases: revenue,
            net_income, gross_profit, gross_margin, operating_income,
            operating_margin, net_margin, eps, pe, market_cap, cash, debt, assets,
            equity, roe, cash_flow, free_cash_flow, price, dividend, beta.
            Example: fields=[\"revenue\", \"operating_margin\"] when comparing
            profitability across companies. Omit to get the full response.
    """
    kept, _ = normalize_tickers(ticker, tickers)
    if tickers or len(kept) > 1:
        return financial_data_api_batch_fn(
            tickers=[*([ticker] if ticker else []), *(tickers or [])],
            statement_type=statement_type, period=period, years=years, fields=fields,
        )
    if not ticker:
        return "[financial_data_api] Error: provide `ticker`, or a non-empty `tickers` list."
    return financial_data_api_fn(
        ticker=ticker, statement_type=statement_type,
        period=period, years=years, fields=fields,
    )


@lc_tool
def calculator(
    metric: str,
    values: Optional[list] = None,
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
    """Calculate arithmetic or a financial metric. Pass only the fields the chosen metric needs.

    For ANY plain arithmetic (adding numbers, averaging, etc.) use the
    arithmetic metrics with a `values` list — do NOT misuse cagr for this.

    metric (arithmetic): sum | subtract | multiply | divide | average | percent_change
    metric (financial):  cagr | free_cash_flow | fcf_yield | altman_z_score | debt_to_equity
      | current_ratio | gross_margin | revenue_growth | peg_ratio
      | enterprise_value | roe | interest_coverage

    Fields per metric:
      sum / subtract / multiply / divide / average: values (a list of numbers)
        e.g. sum four quarters: metric="sum", values=[7037663,7336125,7418435,7591949]
      percent_change: start_value, end_value (simple % change, not annualized)
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
        values=values,
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


@lc_tool
def fx_rate(from_currency: str, to_currency: str) -> str:
    """FX rate. Only if the user asks to convert; then calculator multiply.
    Args:
        from_currency: ISO code e.g. USD
        to_currency: ISO code e.g. INR
    """
    return fx_rate_fn(from_currency=from_currency, to_currency=to_currency)


ALL_TOOLS = [
    sec_filing_search,
    web_search,
    earnings_transcript,
    financial_data_api,
    calculator,
    fred_series,
    fx_rate,
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