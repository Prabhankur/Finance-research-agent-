"""
database/normalizer.py  –  Turns raw provider responses into typed DB rows.

This is what upgrades the caching layer from "have I seen this exact call
recently" to "here's Microsoft's FY2024 revenue as of the last time we saw
it, with a source_id pointing at the exact retrieval that produced it."

Two supported provider shapes today:
  - Alpha Vantage: JSON with fixed field names ("totalRevenue", "netIncome"
    etc.) — documented at https://www.alphavantage.co/documentation/
  - yfinance: the formatted-text output already produced by
    tools/financial_data.py, parsed regex-style. This is fragile by nature,
    but the tool has been outputting this exact shape for months and it's
    what the whole rest of the project already reads.

BOTH paths land rows in the SAME normalized tables. That means when the
report pipeline (yfinance-first) writes AAPL FY2024 revenue and then chat
(via cache) hits the Alpha Vantage path for the same period, the two
distinct source_ids let a later contradiction-detector node see the
disagreement instead of one silently overwriting the other.

FAILURE POSTURE
Normalization runs AFTER the tool has already returned its text to the
caller. If parsing fails — an unexpected AV field name, a yfinance output
format change — the tool call is unaffected. The row just doesn't get
written; caching still works via response_cache; the next fix patches the
parser. Nothing user-visible breaks.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from database.connection import execute, query_one


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  Company upsert
# ─────────────────────────────────────────────────────────────────────────────

def upsert_company(ticker: str, overview: dict | None = None) -> Optional[int]:
    """Returns the companies.id, creating or updating the row as needed.

    Every fact table has a FK to companies(id), so this needs to run before
    anything else can be inserted. Called on every normalize_* path — no-op
    if the row already exists and no new fields are supplied.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        return None

    now = _now_iso()
    fields = {
        "ticker": ticker,
        "name": (overview or {}).get("Company Name") or (overview or {}).get("Name"),
        "sector": (overview or {}).get("Sector"),
        "industry": (overview or {}).get("Industry"),
        "currency": (overview or {}).get("Currency"),
        "cik": (overview or {}).get("CIK"),
        "website": (overview or {}).get("Website"),
        "country": (overview or {}).get("Country"),
    }
    # Drop Nones so an "overview" call with limited fields doesn't wipe out
    # richer data from an earlier call — partial info shouldn't be able to
    # regress existing rows.
    fields = {k: v for k, v in fields.items() if v is not None}

    try:
        existing = query_one("SELECT id FROM companies WHERE ticker = ?", (ticker,))
        if existing:
            company_id = existing["id"]
            if fields:
                sets = ", ".join(f"{k} = ?" for k in fields.keys())
                execute(
                    f"UPDATE companies SET {sets}, updated_at = ? WHERE id = ?",
                    tuple(fields.values()) + (now, company_id),
                )
            return company_id

        # Insert. `ticker` is guaranteed present; the rest fill NULLs where
        # the source didn't have that field.
        all_cols = ["ticker", "name", "sector", "industry", "currency", "cik",
                    "website", "country", "created_at", "updated_at"]
        values = tuple(fields.get(c) for c in all_cols[:-2]) + (now, now)
        placeholders = ", ".join("?" for _ in all_cols)
        return execute(
            f"INSERT INTO companies ({', '.join(all_cols)}) VALUES ({placeholders})",
            values,
        )
    except Exception as exc:
        print(f"[Normalizer] upsert_company({ticker}) failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Source recording
# ─────────────────────────────────────────────────────────────────────────────

def record_source(provider: str, source_type: str = "api", url: str | None = None) -> Optional[int]:
    """Inserts a sources row and returns its id — every fact row FK's to it."""
    try:
        return execute(
            """
            INSERT INTO sources (source_type, provider, url, retrieved_at)
            VALUES (?, ?, ?, ?)
            """,
            (source_type, provider, url, _now_iso()),
        )
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Numeric parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_float(value: Any) -> Optional[float]:
    """Coerces AV's stringified numbers and 'None' sentinels to real floats."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _period_type(fiscal_period: str) -> str:
    """AV's fiscalDateEnding is just a date; we infer from context."""
    return fiscal_period.upper() if fiscal_period.upper() in ("Q1", "Q2", "Q3", "Q4") else "annual"


# ─────────────────────────────────────────────────────────────────────────────
#  Alpha Vantage normalization
# ─────────────────────────────────────────────────────────────────────────────

# AV's endpoint field names, kept explicit rather than assumed. Their docs
# are stable but the field-naming quirks (e.g. "totalRevenue" vs
# "revenueFromContractWithCustomerExcludingAssessedTax") are exactly the
# thing that has bit us before elsewhere in the project — mapping them once
# here means fixes land in one place.
_AV_INCOME_MAP = {
    "revenue":          "totalRevenue",
    "cost_of_revenue":  "costOfRevenue",
    "gross_profit":     "grossProfit",
    "operating_income": "operatingIncome",
    "net_income":       "netIncome",
    "eps_basic":        None,  # not in AV income statement; from OVERVIEW instead
    "ebitda":           "ebitda",
    "rd_expense":       "researchAndDevelopment",
}

_AV_BALANCE_MAP = {
    "cash":                "cashAndCashEquivalentsAtCarryingValue",
    "total_assets":        "totalAssets",
    "total_liabilities":   "totalLiabilities",
    "total_equity":        "totalShareholderEquity",
    "total_debt":          "shortLongTermDebtTotal",
    "current_assets":      "totalCurrentAssets",
    "current_liabilities": "totalCurrentLiabilities",
    "retained_earnings":   "retainedEarnings",
}

_AV_CASHFLOW_MAP = {
    "operating_cash_flow": "operatingCashflow",
    "capital_expenditure": "capitalExpenditures",
    "investing_cash_flow": "cashflowFromInvestment",
    "financing_cash_flow": "cashflowFromFinancing",
}


def _pick(row: dict, mapping: dict, period_end: str) -> dict:
    """Extracts and coerces the fields we care about from an AV row."""
    out: dict[str, Any] = {"period_end": period_end}
    for our_name, av_name in mapping.items():
        if av_name is None:
            continue
        out[our_name] = _to_float(row.get(av_name))
    return out


def normalize_av_income(ticker: str, payload: dict, source_id: int | None) -> int:
    """Writes AV INCOME_STATEMENT annual+quarterly reports to income_statements."""
    company_id = upsert_company(ticker)
    if not company_id:
        return 0

    written = 0
    for period_type, key in (("annual", "annualReports"), ("quarterly", "quarterlyReports")):
        for row in payload.get(key, []) or []:
            period_end = row.get("fiscalDateEnding")
            if not period_end:
                continue
            fields = _pick(row, _AV_INCOME_MAP, period_end)
            fields.update({
                "company_id": company_id,
                "period_type": period_type,
                "currency": row.get("reportedCurrency"),
                "source_id": source_id,
                "updated_at": _now_iso(),
            })
            if _upsert_income(fields):
                written += 1
    return written


def _upsert_income(f: dict) -> bool:
    """One row per (company, period_end, period_type, source_id) — the UNIQUE
    constraint from schema.py. Using INSERT OR REPLACE (portable to Postgres
    via the ON CONFLICT translation) keeps this idempotent so re-running the
    same call doesn't multiply rows."""
    try:
        execute(
            """
            INSERT INTO income_statements
                (company_id, fiscal_period, period_end, period_type,
                 revenue, cost_of_revenue, gross_profit, operating_income,
                 net_income, eps_basic, eps_diluted, ebitda, rd_expense,
                 currency, source_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(company_id, period_end, period_type, source_id) DO UPDATE SET
                revenue = excluded.revenue,
                gross_profit = excluded.gross_profit,
                operating_income = excluded.operating_income,
                net_income = excluded.net_income,
                ebitda = excluded.ebitda,
                rd_expense = excluded.rd_expense,
                updated_at = excluded.updated_at
            """,
            (
                f["company_id"], f.get("fiscal_period"), f["period_end"], f["period_type"],
                f.get("revenue"), f.get("cost_of_revenue"), f.get("gross_profit"),
                f.get("operating_income"), f.get("net_income"),
                f.get("eps_basic"), f.get("eps_diluted"),
                f.get("ebitda"), f.get("rd_expense"),
                f.get("currency"), f.get("source_id"), f["updated_at"],
            ),
        )
        return True
    except Exception as exc:
        print(f"[Normalizer] income upsert failed: {exc}")
        return False


def normalize_av_balance(ticker: str, payload: dict, source_id: int | None) -> int:
    company_id = upsert_company(ticker)
    if not company_id:
        return 0
    written = 0
    for period_type, key in (("annual", "annualReports"), ("quarterly", "quarterlyReports")):
        for row in payload.get(key, []) or []:
            period_end = row.get("fiscalDateEnding")
            if not period_end:
                continue
            f = _pick(row, _AV_BALANCE_MAP, period_end)
            f.update({
                "company_id": company_id,
                "period_type": period_type,
                "currency": row.get("reportedCurrency"),
                "source_id": source_id,
                "updated_at": _now_iso(),
            })
            try:
                execute(
                    """
                    INSERT INTO balance_sheets
                        (company_id, period_end, period_type, cash, total_assets,
                         total_liabilities, total_equity, total_debt,
                         current_assets, current_liabilities, retained_earnings,
                         currency, source_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(company_id, period_end, period_type, source_id) DO UPDATE SET
                        cash = excluded.cash,
                        total_assets = excluded.total_assets,
                        total_liabilities = excluded.total_liabilities,
                        total_equity = excluded.total_equity,
                        total_debt = excluded.total_debt,
                        current_assets = excluded.current_assets,
                        current_liabilities = excluded.current_liabilities,
                        retained_earnings = excluded.retained_earnings,
                        updated_at = excluded.updated_at
                    """,
                    (
                        f["company_id"], f["period_end"], f["period_type"],
                        f.get("cash"), f.get("total_assets"), f.get("total_liabilities"),
                        f.get("total_equity"), f.get("total_debt"),
                        f.get("current_assets"), f.get("current_liabilities"),
                        f.get("retained_earnings"), f.get("currency"), f.get("source_id"),
                        f["updated_at"],
                    ),
                )
                written += 1
            except Exception as exc:
                print(f"[Normalizer] balance upsert failed: {exc}")
    return written


def normalize_av_cashflow(ticker: str, payload: dict, source_id: int | None) -> int:
    company_id = upsert_company(ticker)
    if not company_id:
        return 0
    written = 0
    for period_type, key in (("annual", "annualReports"), ("quarterly", "quarterlyReports")):
        for row in payload.get(key, []) or []:
            period_end = row.get("fiscalDateEnding")
            if not period_end:
                continue
            f = _pick(row, _AV_CASHFLOW_MAP, period_end)
            ocf = f.get("operating_cash_flow")
            capex = f.get("capital_expenditure")
            # Compute FCF here so consumers don't have to. CapEx is reported
            # as a positive by AV even though it's a cash OUTFLOW — subtract
            # the absolute value.
            fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None
            try:
                execute(
                    """
                    INSERT INTO cash_flow_statements
                        (company_id, period_end, period_type,
                         operating_cash_flow, capital_expenditure,
                         investing_cash_flow, financing_cash_flow, free_cash_flow,
                         currency, source_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(company_id, period_end, period_type, source_id) DO UPDATE SET
                        operating_cash_flow = excluded.operating_cash_flow,
                        capital_expenditure = excluded.capital_expenditure,
                        free_cash_flow = excluded.free_cash_flow,
                        updated_at = excluded.updated_at
                    """,
                    (
                        company_id, period_end, period_type,
                        ocf, capex, f.get("investing_cash_flow"),
                        f.get("financing_cash_flow"), fcf,
                        row.get("reportedCurrency"), source_id, _now_iso(),
                    ),
                )
                written += 1
            except Exception as exc:
                print(f"[Normalizer] cashflow upsert failed: {exc}")
    return written


def normalize_av_overview(ticker: str, payload: dict, source_id: int | None) -> Optional[int]:
    """Writes AV OVERVIEW into companies + valuation_snapshots (multiples,
    which are point-in-time)."""
    overview_for_company = {
        "Company Name": payload.get("Name"),
        "Sector":       payload.get("Sector"),
        "Industry":     payload.get("Industry"),
        "Currency":     payload.get("Currency"),
        "CIK":          payload.get("CIK"),
        "Country":      payload.get("Country"),
    }
    company_id = upsert_company(ticker, overview_for_company)
    if not company_id:
        return 0

    try:
        execute(
            """
            INSERT INTO valuation_snapshots
                (company_id, date, market_cap, pe, forward_pe, price_to_sales,
                 price_to_book, ev_ebitda, peg, dividend_yield, beta, source_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(company_id, date) DO UPDATE SET
                market_cap = excluded.market_cap,
                pe = excluded.pe,
                forward_pe = excluded.forward_pe,
                price_to_sales = excluded.price_to_sales,
                price_to_book = excluded.price_to_book,
                ev_ebitda = excluded.ev_ebitda,
                peg = excluded.peg,
                dividend_yield = excluded.dividend_yield,
                beta = excluded.beta,
                source_id = excluded.source_id
            """,
            (
                company_id, datetime.now(timezone.utc).date().isoformat(),
                _to_float(payload.get("MarketCapitalization")),
                _to_float(payload.get("TrailingPE")),
                _to_float(payload.get("ForwardPE")),
                _to_float(payload.get("PriceToSalesRatioTTM")),
                _to_float(payload.get("PriceToBookRatio")),
                _to_float(payload.get("EVToEBITDA")),
                _to_float(payload.get("PEGRatio")),
                _to_float(payload.get("DividendYield")),
                _to_float(payload.get("Beta")),
                source_id,
            ),
        )
        return 1
    except Exception as exc:
        print(f"[Normalizer] overview upsert failed: {exc}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
#  yfinance text normalization
# ─────────────────────────────────────────────────────────────────────────────

_FIN_TEXT_LINE = re.compile(r"^\s*([A-Za-z][\w &()/-]+?)\s*:\s*([\-\d.,eE]+|None|N/A)\s*$")
_FIN_TEXT_PERIOD = re.compile(r"Period:\s*(\d{4}-\d{2}-\d{2})")

# Text-line field name -> our column. Both yfinance's formatted output and
# any provider that follows the same "Total Revenue: 123" style land here.
_TEXT_TO_COLUMN = {
    "Total Revenue":       ("income",   "revenue"),
    "Gross Profit":        ("income",   "gross_profit"),
    "Operating Income":    ("income",   "operating_income"),
    "Net Income":          ("income",   "net_income"),
    "EPS (Basic)":         ("income",   "eps_basic"),
    "EPS (Diluted)":       ("income",   "eps_diluted"),
    "EBITDA":              ("income",   "ebitda"),
    "R&D Expense":         ("income",   "rd_expense"),

    "Total Assets":        ("balance",  "total_assets"),
    "Total Liabilities":   ("balance",  "total_liabilities"),
    "Shareholder Equity":  ("balance",  "total_equity"),
    "Cash & Equivalents":  ("balance",  "cash"),
    "Total Debt":          ("balance",  "total_debt"),
    "Retained Earnings":   ("balance",  "retained_earnings"),

    "Operating Cash Flow": ("cashflow", "operating_cash_flow"),
    "CapEx":               ("cashflow", "capital_expenditure"),
    "Free Cash Flow":      ("cashflow", "free_cash_flow"),
}


def normalize_yfinance_text(ticker: str, text: str, source_id: int | None,
                            period_type: str = "annual") -> dict:
    """Parses tools/financial_data.py's formatted text output into DB rows.

    Regex-based on purpose: the alternative — vendoring yfinance's data
    model — pulls in a big dependency for what's a stable string format
    the project has been emitting for months. If yfinance's output format
    changes, this parser produces no rows (silently), the tool call still
    works, and it gets patched here in one place.
    """
    company_id = upsert_company(ticker)
    if not company_id:
        return {"income": 0, "balance": 0, "cashflow": 0}

    counts = {"income": 0, "balance": 0, "cashflow": 0}

    # Walk the text period-by-period. yfinance blocks are shaped:
    #   Period: 2026-06-30
    #   Total Revenue : 331839000000
    #   ...
    #   Period: 2025-06-30
    #   ...
    current_period: Optional[str] = None
    by_period: dict[str, dict] = {}

    for line in text.splitlines():
        period_match = _FIN_TEXT_PERIOD.search(line)
        if period_match:
            current_period = period_match.group(1)
            by_period.setdefault(current_period, {"income": {}, "balance": {}, "cashflow": {}})
            continue

        line_match = _FIN_TEXT_LINE.match(line)
        if not line_match or not current_period:
            continue

        key, raw_value = line_match.group(1).strip(), line_match.group(2).strip()
        target = _TEXT_TO_COLUMN.get(key)
        if not target:
            continue

        section, column = target
        by_period[current_period][section][column] = _to_float(raw_value)

    now = _now_iso()

    for period_end, sections in by_period.items():
        if sections["income"]:
            f = sections["income"]
            f.update({
                "company_id": company_id, "fiscal_period": None,
                "period_end": period_end, "period_type": period_type,
                "source_id": source_id, "updated_at": now,
            })
            if _upsert_income(f):
                counts["income"] += 1

        if sections["balance"]:
            b = sections["balance"]
            try:
                execute(
                    """
                    INSERT INTO balance_sheets
                        (company_id, period_end, period_type, cash, total_assets,
                         total_liabilities, total_equity, total_debt,
                         retained_earnings, source_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(company_id, period_end, period_type, source_id) DO UPDATE SET
                        cash = excluded.cash,
                        total_assets = excluded.total_assets,
                        total_liabilities = excluded.total_liabilities,
                        total_equity = excluded.total_equity,
                        total_debt = excluded.total_debt,
                        retained_earnings = excluded.retained_earnings,
                        updated_at = excluded.updated_at
                    """,
                    (
                        company_id, period_end, period_type,
                        b.get("cash"), b.get("total_assets"), b.get("total_liabilities"),
                        b.get("total_equity"), b.get("total_debt"),
                        b.get("retained_earnings"), source_id, now,
                    ),
                )
                counts["balance"] += 1
            except Exception as exc:
                print(f"[Normalizer] yfinance balance upsert failed: {exc}")

        if sections["cashflow"]:
            c = sections["cashflow"]
            ocf = c.get("operating_cash_flow")
            capex = c.get("capital_expenditure")
            fcf = c.get("free_cash_flow")
            if fcf is None and ocf is not None and capex is not None:
                fcf = ocf - abs(capex)
            try:
                execute(
                    """
                    INSERT INTO cash_flow_statements
                        (company_id, period_end, period_type,
                         operating_cash_flow, capital_expenditure, free_cash_flow,
                         source_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(company_id, period_end, period_type, source_id) DO UPDATE SET
                        operating_cash_flow = excluded.operating_cash_flow,
                        capital_expenditure = excluded.capital_expenditure,
                        free_cash_flow = excluded.free_cash_flow,
                        updated_at = excluded.updated_at
                    """,
                    (company_id, period_end, period_type, ocf, capex, fcf, source_id, now),
                )
                counts["cashflow"] += 1
            except Exception as exc:
                print(f"[Normalizer] yfinance cashflow upsert failed: {exc}")

    return counts


# ─────────────────────────────────────────────────────────────────────────────
#  SEC XBRL text normalization
# ─────────────────────────────────────────────────────────────────────────────
# Structurally different from yfinance's text format, which is why this is a
# separate parser rather than reusing normalize_yfinance_text:
#   - yfinance:  "Period: 2024-06-30" header, then raw integers per line
#   - SEC XBRL:  scaled human-readable values PER LINE, with the period_end
#                embedded inline: "  Total Revenue  :  $391.04B  (FY end: 2024-09-28)"
# So parsing needs to (a) un-scale "$391.04B" back to a raw float, and
# (b) extract period_end from the same line rather than a shared header.

_SEC_LINE_RE = re.compile(
    r"^\s*(.+?)\s*:\s*\$?([\d,]+\.?\d*)\s*([BMK])?\s*"
    r"\((?:FY|Q)\s*end:\s*(\d{4}-\d{2}-\d{2})\)\s*$"
)

# SEC's line labels (from tools/sec_filing.py's _XBRL_CONCEPTS dict values)
# differ slightly from yfinance's — "Shareholders Equity" not "Shareholder
# Equity", "Long-Term Debt" not "Total Debt". Mapped explicitly rather than
# assumed identical, which is exactly the kind of silent mismatch that has
# bitten this project before (see the gross_margin duplicate-schema bug).
_SEC_LABEL_MAP: dict[str, tuple[str, str]] = {
    "Total Revenue":       ("income", "revenue"),
    "Net Income":          ("income", "net_income"),
    "EPS (Diluted)":       ("income", "eps_diluted"),
    "Gross Profit":        ("income", "gross_profit"),
    "Operating Income":    ("income", "operating_income"),
    "R&D Expense":         ("income", "rd_expense"),
    "Cash & Equivalents":  ("balance", "cash"),
    "Long-Term Debt":      ("balance", "total_debt"),
    "Shareholders Equity": ("balance", "total_equity"),
}

_SEC_SCALE = {"B": 1e9, "M": 1e6, "K": 1e3, None: 1.0}


def normalize_sec_xbrl_text(
    ticker: str, text: str, source_id: int | None, period_type: str = "annual"
) -> dict:
    """
    Parses tools/sec_filing.py's XBRL rendering into the SAME
    income_statements / balance_sheets tables yfinance/AV write to — SEC is
    tier A, so when the contradiction detector compares this against a
    tier-B figure for the same (company, period_end, field), it's exactly
    the kind of disagreement this project was built to surface, not hide.

    Lines that say "N/A" or "N/A for FYxxxx" (see sec_filing.py's honest
    unavailable-year handling) don't match _SEC_LINE_RE at all and are
    silently skipped here — correct behaviour, there's nothing to store.
    """
    company_id = upsert_company(ticker)
    if not company_id:
        return {"income": 0, "balance": 0}

    by_period: dict[str, dict[str, dict]] = {}

    for line in text.splitlines():
        match = _SEC_LINE_RE.match(line)
        if not match:
            continue
        label, raw_number, scale_letter, period_end = match.groups()
        target = _SEC_LABEL_MAP.get(label.strip())
        if not target:
            continue

        try:
            value = float(raw_number.replace(",", "")) * _SEC_SCALE.get(scale_letter)
        except (TypeError, ValueError):
            continue

        section, column = target
        by_period.setdefault(period_end, {"income": {}, "balance": {}})
        by_period[period_end][section][column] = value

    counts = {"income": 0, "balance": 0}
    now = _now_iso()

    for period_end, sections in by_period.items():
        if sections["income"]:
            f = dict(sections["income"])
            f.update({
                "company_id": company_id, "fiscal_period": None,
                "period_end": period_end, "period_type": period_type,
                "source_id": source_id, "updated_at": now,
            })
            if _upsert_income(f):
                counts["income"] += 1

        if sections["balance"]:
            b = sections["balance"]
            try:
                execute(
                    """
                    INSERT INTO balance_sheets
                        (company_id, period_end, period_type, cash, total_debt,
                         total_equity, source_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(company_id, period_end, period_type, source_id) DO UPDATE SET
                        cash = excluded.cash,
                        total_debt = excluded.total_debt,
                        total_equity = excluded.total_equity,
                        updated_at = excluded.updated_at
                    """,
                    (
                        company_id, period_end, period_type,
                        b.get("cash"), b.get("total_debt"), b.get("total_equity"),
                        source_id, now,
                    ),
                )
                counts["balance"] += 1
            except Exception as exc:
                print(f"[Normalizer] SEC balance upsert failed: {exc}")

    return counts


# ─────────────────────────────────────────────────────────────────────────────
#  FRED normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_fred_series(series_id: str, name: str | None, text: str,
                          source_id: int | None) -> int:
    """Parses tools/fred.py's "Date  Value" table output into macro_observations."""
    try:
        series_row = query_one("SELECT id FROM macro_series WHERE series_code = ?", (series_id,))
        if series_row:
            series_pk = series_row["id"]
        else:
            series_pk = execute(
                """INSERT INTO macro_series (series_code, name, source, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (series_id, name, "FRED", _now_iso()),
            )
    except Exception as exc:
        print(f"[Normalizer] macro_series upsert failed: {exc}")
        return 0

    written = 0
    row_re = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s+([\-\d.]+)\s*$")
    for line in text.splitlines():
        match = row_re.match(line)
        if not match:
            continue
        date, value = match.group(1), _to_float(match.group(2))
        if value is None:
            continue
        try:
            execute(
                """
                INSERT INTO macro_observations (series_id, date, value, source_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(series_id, date) DO UPDATE SET
                    value = excluded.value, source_id = excluded.source_id
                """,
                (series_pk, date, value, source_id),
            )
            written += 1
        except Exception as exc:
            print(f"[Normalizer] macro_observation upsert failed: {exc}")

    return written
