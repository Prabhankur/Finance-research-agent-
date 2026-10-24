"""
database/financial_reads.py  –  Reconstructs a tool response from typed rows.

The other half of normalization. Once tool responses land as typed rows in
income_statements / balance_sheets / cash_flow_statements (see
database/normalizer.py), this module answers the reverse question: given a
`(ticker, period_type, years)` request, is there enough fresh data in the DB
to synthesize the same text response WITHOUT any API call?

Layered on top of response_cache rather than replacing it. Order for a
financial_data_api request now goes:

  1. Text-cache hit? (fastest — same wire response we sent before)  → return
  2. Fresh normalized rows for this ticker + period_type?           → rebuild
  3. Live yfinance/AV                                               → return + normalize + cache

Only #2 is new. It matters for a real workload pattern: chat asks about MSFT
revenue, then a report a day later needs MSFT's income statement. Under the
old text-cache-only design, that report's call was a MISS (different call
signature — different `years` or `statement_type`) even though the DB had
every number it needed. This module closes that gap.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from database.connection import query, query_one


# Same TTL policy as caching (see schema.TTL_SECONDS), applied at read time
# instead of write time. Rows older than these are ignored — the API gets
# called instead, and the row is refreshed on the way out.
_MAX_AGE = {
    "annual":    timedelta(days=7),
    "quarterly": timedelta(days=1),
}


def _fresh_enough(updated_at: str, period_type: str) -> bool:
    try:
        moment = datetime.fromisoformat(updated_at)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
    except Exception:
        return False
    age = datetime.now(timezone.utc) - moment
    return age <= _MAX_AGE.get(period_type, timedelta(days=7))


def _get_company_id(ticker: str) -> Optional[int]:
    row = query_one("SELECT id FROM companies WHERE ticker = ?", (ticker.upper(),))
    return row["id"] if row else None


def _fmt_num(value) -> str:
    """Renders a stored float back to the same integer-style yfinance uses.
    Kept identical to tools/financial_data.py so a rebuilt response is
    byte-for-byte indistinguishable from a live one — the LLM sees no
    difference between the two paths."""
    if value is None:
        return "N/A"
    try:
        return f"{int(float(value))}"
    except Exception:
        return str(value)


def try_rebuild_financials(
    ticker: str, statement_type: str, period_type: str, years: int
) -> Optional[str]:
    """
    Returns a reconstructed text response if the DB has enough fresh data,
    else None (which is the signal to fall through to the live API).

    Only handles the "income/balance/cashflow" statement types today —
    "overview" and "ratios" are point-in-time snapshots kept in
    valuation_snapshots, whose freshness policy is much stricter (1h),
    and rebuilding them adds complexity the report pipeline barely uses.
    Not a limitation of the design, just scoped to what's high-value now.
    """
    if statement_type not in ("income", "balance", "cashflow", "all"):
        return None

    company_id = _get_company_id(ticker)
    if not company_id:
        return None

    sections: list[str] = []

    if statement_type in ("income", "all"):
        rows = query(
            """SELECT period_end, revenue, gross_profit, operating_income,
                      net_income, eps_basic, ebitda, rd_expense, updated_at, currency
               FROM income_statements
               WHERE company_id = ? AND period_type = ?
               ORDER BY period_end DESC LIMIT ?""",
            (company_id, period_type, years),
        )
        if len(rows) < years or not all(_fresh_enough(r["updated_at"], period_type) for r in rows):
            return None
        sections.append(_render_income(ticker, period_type, rows))

    if statement_type in ("balance", "all"):
        rows = query(
            """SELECT period_end, cash, total_assets, total_liabilities, total_equity,
                      total_debt, retained_earnings, updated_at
               FROM balance_sheets
               WHERE company_id = ? AND period_type = ?
               ORDER BY period_end DESC LIMIT ?""",
            (company_id, period_type, years),
        )
        if len(rows) < years or not all(_fresh_enough(r["updated_at"], period_type) for r in rows):
            return None
        sections.append(_render_balance(ticker, period_type, rows))

    if statement_type in ("cashflow", "all"):
        rows = query(
            """SELECT period_end, operating_cash_flow, capital_expenditure,
                      free_cash_flow, updated_at
               FROM cash_flow_statements
               WHERE company_id = ? AND period_type = ?
               ORDER BY period_end DESC LIMIT ?""",
            (company_id, period_type, years),
        )
        if len(rows) < years or not all(_fresh_enough(r["updated_at"], period_type) for r in rows):
            return None
        sections.append(_render_cashflow(ticker, period_type, rows))

    if not sections:
        return None

    # Marker in the header so a human reading the raw output can tell it
    # came from the local DB rather than a live provider — same content,
    # just visibly attributed. The LLM sees it too; it's fine.
    header = f"[FinData:DB] {statement_type} ({period_type}) – {ticker}\n"
    return header + "\n\n".join(sections)


def _render_income(ticker: str, period_type: str, rows: list[dict]) -> str:
    label = "Income Statement (Annual)" if period_type == "annual" else "Income Statement (Quarterly)"
    out = [f"[FinData] {label} – {ticker}"]
    for row in rows:
        out.append(f"Period: {row['period_end']}")
        out.append(f"  Total Revenue      : {_fmt_num(row.get('revenue'))}")
        out.append(f"  Gross Profit       : {_fmt_num(row.get('gross_profit'))}")
        out.append(f"  Operating Income   : {_fmt_num(row.get('operating_income'))}")
        out.append(f"  Net Income         : {_fmt_num(row.get('net_income'))}")
        if row.get("eps_basic") is not None:
            out.append(f"  EPS (Basic)        : {row['eps_basic']}")
        if row.get("ebitda") is not None:
            out.append(f"  EBITDA             : {_fmt_num(row.get('ebitda'))}")
        if row.get("rd_expense") is not None:
            out.append(f"  R&D Expense        : {_fmt_num(row.get('rd_expense'))}")
    return "\n".join(out)


def _render_balance(ticker: str, period_type: str, rows: list[dict]) -> str:
    label = "Balance Sheet (Annual)" if period_type == "annual" else "Balance Sheet (Quarterly)"
    out = [f"[FinData] {label} – {ticker}"]
    for row in rows:
        out.append(f"Period: {row['period_end']}")
        out.append(f"  Total Assets       : {_fmt_num(row.get('total_assets'))}")
        out.append(f"  Total Liabilities  : {_fmt_num(row.get('total_liabilities'))}")
        out.append(f"  Shareholder Equity : {_fmt_num(row.get('total_equity'))}")
        out.append(f"  Cash & Equivalents : {_fmt_num(row.get('cash'))}")
        out.append(f"  Total Debt         : {_fmt_num(row.get('total_debt'))}")
        if row.get("retained_earnings") is not None:
            out.append(f"  Retained Earnings  : {_fmt_num(row.get('retained_earnings'))}")
    return "\n".join(out)


def _render_cashflow(ticker: str, period_type: str, rows: list[dict]) -> str:
    label = "Cash Flow (Annual)" if period_type == "annual" else "Cash Flow (Quarterly)"
    out = [f"[FinData] {label} – {ticker}"]
    for row in rows:
        out.append(f"Period: {row['period_end']}")
        out.append(f"  Operating Cash Flow: {_fmt_num(row.get('operating_cash_flow'))}")
        out.append(f"  CapEx              : {_fmt_num(row.get('capital_expenditure'))}")
        out.append(f"  Free Cash Flow     : {_fmt_num(row.get('free_cash_flow'))}")
    return "\n".join(out)
