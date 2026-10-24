"""
tools/calculator.py  –  Financial calculator for custom indices and derived metrics.

The agent calls this whenever it needs to compute something that isn't
directly returned by an API – e.g. Free Cash Flow Yield, Altman Z-Score,
custom growth rates, CAGR, debt coverage ratios, etc.
"""

from __future__ import annotations
import math
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  Core calculation functions
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v: Any, default: float = 0.0) -> float:
    """Convert value to float safely, returning default on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def cagr(start_value: float, end_value: float, years: float) -> str:
    """Compound Annual Growth Rate."""
    if start_value <= 0 or years <= 0:
        return "Error: start_value and years must be positive."
    result = (end_value / start_value) ** (1 / years) - 1
    return f"CAGR = {result * 100:.2f}%"


def free_cash_flow(operating_cash_flow: float, capex: float) -> str:
    """Free Cash Flow = Operating Cash Flow - Capital Expenditures."""
    fcf = _safe(operating_cash_flow) - abs(_safe(capex))
    return f"Free Cash Flow = ${fcf:,.0f}"


def fcf_yield(fcf: float, market_cap: float) -> str:
    """FCF Yield = Free Cash Flow / Market Cap."""
    if market_cap == 0:
        return "Error: Market cap cannot be zero."
    result = _safe(fcf) / _safe(market_cap) * 100
    return f"FCF Yield = {result:.2f}%"


def altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
    total_assets: float,
) -> str:
    """
    Altman Z-Score for public companies.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    Interpretation:
        Z > 2.99  → Safe zone
        1.81–2.99 → Grey zone
        Z < 1.81  → Distress zone
    """
    ta = _safe(total_assets)
    if ta == 0:
        return "Error: total_assets cannot be zero."

    x1 = _safe(working_capital) / ta
    x2 = _safe(retained_earnings) / ta
    x3 = _safe(ebit) / ta
    x4 = _safe(market_cap) / max(_safe(total_liabilities), 1)
    x5 = _safe(revenue) / ta

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        zone = "Safe Zone (financially healthy)"
    elif z >= 1.81:
        zone = "Grey Zone (moderate risk)"
    else:
        zone = "Distress Zone (high bankruptcy risk)"

    return (
        f"Altman Z-Score = {z:.2f} → {zone}\n"
        f"  X1 (Working Capital/TA)       = {x1:.4f}\n"
        f"  X2 (Retained Earnings/TA)     = {x2:.4f}\n"
        f"  X3 (EBIT/TA)                  = {x3:.4f}\n"
        f"  X4 (Market Cap/Total Liab)    = {x4:.4f}\n"
        f"  X5 (Revenue/TA)               = {x5:.4f}"
    )


def debt_to_equity(total_debt: float, shareholder_equity: float) -> str:
    if shareholder_equity == 0:
        return "Error: shareholder_equity cannot be zero."
    result = _safe(total_debt) / _safe(shareholder_equity)
    return f"Debt/Equity Ratio = {result:.2f}"


def current_ratio(current_assets: float, current_liabilities: float) -> str:
    if current_liabilities == 0:
        return "Error: current_liabilities cannot be zero."
    result = _safe(current_assets) / _safe(current_liabilities)
    return f"Current Ratio = {result:.2f}"


def gross_margin(revenue: float, cogs: float = None, gross_profit: float = None) -> str:
    """
    Gross Margin = Gross Profit / Revenue.

    Accepts EITHER cogs OR gross_profit, not just cogs. This existed as a
    real production bug: yfinance/Alpha Vantage return gross_profit directly
    on the income statement — they don't return cost-of-goods-sold as its own
    line. The chat agent, holding a real gross_profit figure in context,
    naturally tried to pass it straight through, but the old signature only
    recognized `cogs`, silently dropped the unrecognized key, and crashed
    with "missing 1 required positional argument: cogs" — even though the
    agent had the exact number needed, just under the tool's unexpected name.
    """
    if gross_profit is None:
        if cogs is None:
            return "Error: provide either cogs or gross_profit."
        gross_profit = _safe(revenue) - _safe(cogs)
    if _safe(revenue) == 0:
        return "Error: revenue cannot be zero."
    result = _safe(gross_profit) / _safe(revenue) * 100
    return f"Gross Margin = {result:.2f}%"
    if revenue == 0:
        return "Error: revenue cannot be zero."
    result = (_safe(revenue) - _safe(cogs)) / _safe(revenue) * 100
    return f"Gross Margin = {result:.2f}%"


def revenue_growth(current_revenue: float, prior_revenue: float) -> str:
    if prior_revenue == 0:
        return "Error: prior_revenue cannot be zero."
    result = (_safe(current_revenue) - _safe(prior_revenue)) / _safe(prior_revenue) * 100
    return f"Revenue Growth = {result:.2f}%"


def peg_ratio(pe_ratio: float, earnings_growth_rate: float) -> str:
    """PEG Ratio = P/E / EPS Growth Rate. Growth rate as percentage, e.g. 15 for 15%."""
    if earnings_growth_rate == 0:
        return "Error: earnings_growth_rate cannot be zero."
    result = _safe(pe_ratio) / _safe(earnings_growth_rate)
    return f"PEG Ratio = {result:.2f} ({'Undervalued' if result < 1 else 'Overvalued' if result > 2 else 'Fairly Valued'})"


def enterprise_value(
    market_cap: float,
    total_debt: float,
    cash: float,
) -> str:
    ev = _safe(market_cap) + _safe(total_debt) - _safe(cash)
    return f"Enterprise Value = ${ev:,.0f}"


def roe(net_income: float, shareholder_equity: float) -> str:
    if shareholder_equity == 0:
        return "Error: shareholder_equity cannot be zero."
    result = _safe(net_income) / _safe(shareholder_equity) * 100
    return f"Return on Equity (ROE) = {result:.2f}%"


def interest_coverage(ebit: float, interest_expense: float) -> str:
    if interest_expense == 0:
        return "Error: interest_expense cannot be zero."
    result = _safe(ebit) / _safe(interest_expense)
    interpretation = (
        "Strong coverage"  if result > 5 else
        "Adequate"         if result > 2 else
        "Risky coverage"
    )
    return f"Interest Coverage Ratio = {result:.2f} ({interpretation})"


# ─────────────────────────────────────────────────────────────────────────────
#  Master dispatcher – the LLM calls THIS single function
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_METRICS = [
    "cagr", "free_cash_flow", "fcf_yield", "altman_z_score",
    "debt_to_equity", "current_ratio", "gross_margin", "revenue_growth",
    "peg_ratio", "enterprise_value", "roe", "interest_coverage",
]


_DISPATCH = {
    "cagr":               cagr,
    "free_cash_flow":     free_cash_flow,
    "fcf_yield":          fcf_yield,
    "altman_z_score":     altman_z_score,
    "debt_to_equity":     debt_to_equity,
    "current_ratio":      current_ratio,
    "gross_margin":       gross_margin,
    "revenue_growth":     revenue_growth,
    "peg_ratio":          peg_ratio,
    "enterprise_value":   enterprise_value,
    "roe":                roe,
    "interest_coverage":  interest_coverage,
}

# Which of the flat parameters below each metric actually needs. Used to
# filter out the unused None-valued params before calling the real function,
# and to tell a caller (e.g. the compute executor / replanner) exactly what
# a given metric requires without it having to guess from a docstring.
METRIC_REQUIRED_ARGS = {
    "cagr":               ["start_value", "end_value", "years"],
    "free_cash_flow":     ["operating_cash_flow", "capex"],
    "fcf_yield":          ["fcf", "market_cap"],
    "altman_z_score":     ["working_capital", "retained_earnings", "ebit",
                            "market_cap", "total_liabilities", "revenue", "total_assets"],
    "debt_to_equity":     ["total_debt", "shareholder_equity"],
    "current_ratio":      ["current_assets", "current_liabilities"],
    "gross_margin":       ["revenue", "cogs"],
    "revenue_growth":     ["current_revenue", "prior_revenue"],
    "peg_ratio":          ["pe_ratio", "earnings_growth_rate"],
    "enterprise_value":   ["market_cap", "total_debt", "cash"],
    "roe":                ["net_income", "shareholder_equity"],
    "interest_coverage":  ["ebit", "interest_expense"],
}


def _calculate(metric: str, **kwargs) -> str:
    """
    Master dispatcher for all financial calculations. Internal — kept
    accepting **kwargs so it's still a plain, easily-testable Python function.
    The LLM-facing tool wrapper (calculator, below) never uses **kwargs in
    its own signature, because LangChain's @tool decorator can't turn a
    **kwargs parameter into real named fields in the JSON schema it shows
    the model — it collapses to a single opaque 'kwargs' object, which is
    exactly what was causing every call to fail before this fix.
    """
    metric = metric.lower().strip()

    if metric not in _DISPATCH:
        return (
            f"[Calculator] Unknown metric '{metric}'. "
            f"Supported: {', '.join(SUPPORTED_METRICS)}"
        )

    required = list(METRIC_REQUIRED_ARGS[metric])

    # gross_margin special case: gross_profit is an accepted alternative to
    # cogs (see gross_margin()'s docstring — providers return gross profit
    # directly, not COGS). Swap it into what actually gets forwarded, and
    # stop the generic "missing cogs" check from firing when gross_profit
    # was the thing supplied instead.
    extra_allowed: list[str] = []
    if metric == "gross_margin" and kwargs.get("gross_profit") is not None:
        extra_allowed = ["gross_profit"]
        required = [a for a in required if a != "cogs"]

    # Only pass through args this specific metric actually needs, dropping
    # anything that wasn't supplied (None) rather than coercing it to 0.0 —
    # a missing input should surface as a clear TypeError, not silently
    # compute a wrong number.
    provided = {
        k: kwargs[k] for k in required + extra_allowed if kwargs.get(k) is not None
    }

    try:
        float_kwargs = {k: _safe(v) for k, v in provided.items()}
        return f"[Calculator] {_DISPATCH[metric](**float_kwargs)}"
    except TypeError as exc:
        missing = [a for a in required if a not in provided]
        detail = f" Missing: {', '.join(missing)}." if missing else ""
        return f"[Calculator] Missing or wrong argument for '{metric}': {exc}.{detail}"
    except Exception as exc:
        return f"[Calculator] Error computing '{metric}': {exc}"


def calculator(
    metric: str,
    start_value: float = None,
    end_value: float = None,
    years: float = None,
    operating_cash_flow: float = None,
    capex: float = None,
    fcf: float = None,
    market_cap: float = None,
    working_capital: float = None,
    retained_earnings: float = None,
    ebit: float = None,
    total_liabilities: float = None,
    revenue: float = None,
    total_assets: float = None,
    total_debt: float = None,
    shareholder_equity: float = None,
    current_assets: float = None,
    current_liabilities: float = None,
    cogs: float = None,
    gross_profit: float = None,
    current_revenue: float = None,
    prior_revenue: float = None,
    pe_ratio: float = None,
    earnings_growth_rate: float = None,
    cash: float = None,
    net_income: float = None,
    interest_expense: float = None,
) -> str:
    """
    Calculates a custom financial metric. Every possible input is a real,
    named, optional parameter (not **kwargs) so the LLM's tool schema shows
    the exact field names for every metric instead of one opaque object.

    metric: one of cagr, free_cash_flow, fcf_yield, altman_z_score,
            debt_to_equity, current_ratio, gross_margin, revenue_growth,
            peg_ratio, enterprise_value, roe, interest_coverage.
    Only pass the arguments relevant to the chosen metric — see
    METRIC_REQUIRED_ARGS for exactly which ones each metric needs.

    gross_margin accepts EITHER cogs OR gross_profit — use whichever your
    data actually has. Most financial-data APIs return gross_profit
    directly; only pass cogs if that's what you were given instead.

    Example:
        calculator(metric="cagr", start_value=100, end_value=196, years=7)
        → "[Calculator] CAGR = 10.10%"
    """
    return _calculate(
        metric,
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
