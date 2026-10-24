"""
tools/financial_data.py  –  Structured financial data retrieval.

Combines the user's existing yfinance snapshot code with Alpha Vantage
for income statement, balance sheet, cash flow, and key ratios.
Returns clean JSON-like strings ready to be appended to agent context.
"""

from __future__ import annotations
import os
import json
import requests
import yfinance as yf
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ── Helper: Alpha Vantage generic fetcher ────────────────────────────────────

import time as _time

# Alpha Vantage free tier: 5 calls/minute. A single financial_data_api("all")
# call makes 4 AV requests (income + balance + cashflow + ratios). Two such
# calls (annual + quarterly) = 8 AV requests, plus earnings_transcript = 9.
# Without pacing, the second half of these get rate-limited and return empty.
_AV_LAST_CALL_TIME = 0.0
_AV_MIN_INTERVAL = 13.0  # seconds between AV calls → max ~4.6 calls/min

def _av_fetch(function: str, ticker: str, extra_params: Optional[dict] = None) -> dict:
    """Internal helper – calls Alpha Vantage with rate-limit pacing."""
    global _AV_LAST_CALL_TIME
    av_key = os.getenv("ALPHA_VANTAGE_KEY", "")

    # Pace calls to stay under the free-tier limit
    elapsed = _time.time() - _AV_LAST_CALL_TIME
    if elapsed < _AV_MIN_INTERVAL:
        wait = _AV_MIN_INTERVAL - elapsed
        print(f"  [AV rate-limit] waiting {wait:.1f}s before next call...")
        _time.sleep(wait)

    params = {"function": function, "symbol": ticker, "apikey": av_key}
    if extra_params:
        params.update(extra_params)
    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
    _AV_LAST_CALL_TIME = _time.time()
    resp.raise_for_status()
    data = resp.json()

    # AV returns {"Note": "...call frequency..."} when rate-limited rather
    # than an empty dict. Detect this and return empty so the caller's
    # empty-check picks it up cleanly.
    if "Note" in data and "call frequency" in data.get("Note", "").lower():
        print(f"  [AV rate-limit] API returned rate-limit notice for {function}")
        return {}

    return data


# ── Main tool function ────────────────────────────────────────────────────────

def financial_data_api(
    ticker: str,
    statement_type: str = "overview",
    period: str = "annual",
    years: int = 3,
) -> str:
    """
    Retrieves structured financial data for a company.

    Args:
        ticker:         Stock ticker, e.g. "AAPL".
        statement_type: One of:
                          "overview"      – market snapshot (yfinance)
                          "income"        – income statement
                          "balance"       – balance sheet
                          "cashflow"      – cash flow statement
                          "ratios"        – key financial ratios
                          "all"           – runs all four above
        period:         "annual" or "quarterly"
        years:          How many periods to retrieve (1-5).

    Returns:
        Formatted string of financial data ready for context injection.
    """
    ticker = ticker.upper().strip()
    results: list[str] = []

    # ── 1. Market Overview (yfinance) ─────────────────────────────────────────
    if statement_type in ("overview", "all"):
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info

            if info and "marketCap" in info:
                overview = {
                    "Company Name":         info.get("longName"),
                    "Sector":               info.get("sector"),
                    "Industry":             info.get("industry"),
                    "Current Price":        info.get("currentPrice") or info.get("regularMarketPrice"),
                    "Currency":             info.get("currency"),
                    "Market Cap":           info.get("marketCap"),
                    "Trailing P/E":         info.get("trailingPE"),
                    "Forward P/E":          info.get("forwardPE"),
                    "EV/EBITDA":            info.get("enterpriseToEbitda"),
                    "Price/Book":           info.get("priceToBook"),
                    "Dividend Yield (%)":   round((info.get("dividendYield") or 0) * 100, 2),
                    "52-Week High":         info.get("fiftyTwoWeekHigh"),
                    "52-Week Low":          info.get("fiftyTwoWeekLow"),
                    "50-Day MA":            info.get("fiftyDayAverage"),
                    "200-Day MA":           info.get("twoHundredDayAverage"),
                    "Beta":                 info.get("beta"),
                    "Revenue (TTM)":        info.get("totalRevenue"),
                    "Net Income (TTM)":     info.get("netIncomeToCommon"),
                    "Profit Margin (%)":    round((info.get("profitMargins") or 0) * 100, 2),
                    "Return on Equity (%)": round((info.get("returnOnEquity") or 0) * 100, 2),
                    "Debt/Equity":          info.get("debtToEquity"),
                    "Business Summary":     (info.get("longBusinessSummary") or "")[:400] + "...",
                }
                results.append(
                    f"[FinData] Market Overview – {ticker}\n"
                    f"{'='*60}\n"
                    f"{json.dumps(overview, indent=2)}"
                )
            else:
                results.append(f"[FinData] No yfinance data found for {ticker}.")
        except Exception as exc:
            results.append(f"[FinData] yfinance error for {ticker}: {exc}")

    # ── 2. Income Statement (yfinance primary, AV fallback) ──────────────────
    if statement_type in ("income", "all"):
        try:
            stock = yf.Ticker(ticker)
            fin = stock.financials if period == "annual" else stock.quarterly_financials
            block = [f"[FinData] Income Statement ({period.title()}) – {ticker}"]

            if fin is not None and not fin.empty:
                # yfinance returns a DataFrame with rows=metrics, columns=periods
                cols = fin.columns[:years]
                for col in cols:
                    date_str = col.strftime("%Y-%m-%d")
                    revenue = fin.loc["Total Revenue"].get(col) if "Total Revenue" in fin.index else None
                    gross = fin.loc["Gross Profit"].get(col) if "Gross Profit" in fin.index else None
                    op_income = fin.loc["Operating Income"].get(col) if "Operating Income" in fin.index else None
                    net_income = fin.loc["Net Income"].get(col) if "Net Income" in fin.index else None
                    ebitda = fin.loc["EBITDA"].get(col) if "EBITDA" in fin.index else None
                    rd = fin.loc["Research And Development"].get(col) if "Research And Development" in fin.index else None
                    eps = fin.loc["Basic EPS"].get(col) if "Basic EPS" in fin.index else None
                    block.append(
                        f"\n  Period: {date_str}\n"
                        f"  Total Revenue      : {int(revenue) if revenue else 'N/A'}\n"
                        f"  Gross Profit       : {int(gross) if gross else 'N/A'}\n"
                        f"  Operating Income   : {int(op_income) if op_income else 'N/A'}\n"
                        f"  Net Income         : {int(net_income) if net_income else 'N/A'}\n"
                        f"  EPS (Basic)        : {eps if eps else 'N/A'}\n"
                        f"  EBITDA             : {int(ebitda) if ebitda else 'N/A'}\n"
                        f"  R&D Expense        : {int(rd) if rd else 'N/A'}\n"
                    )
                results.append("\n".join(block))
            else:
                # yfinance had nothing — try AV as fallback
                raise ValueError("yfinance returned empty financials")
        except Exception as yf_exc:
            # Fallback: Alpha Vantage
            try:
                av_func = "INCOME_STATEMENT"
                data    = _av_fetch(av_func, ticker)
                key     = "annualReports" if period == "annual" else "quarterlyReports"
                reports = data.get(key, [])[:years]
                block   = [f"[FinData] Income Statement ({period.title()}) – {ticker}"]
                if not reports:
                    block.append(f"\n  [Data unavailable] Neither yfinance ({yf_exc}) nor Alpha Vantage returned {period} income statement data for {ticker}.")
                for r in reports:
                    block.append(
                        f"\n  Period: {r.get('fiscalDateEnding')}\n"
                        f"  Total Revenue      : {r.get('totalRevenue')}\n"
                        f"  Gross Profit       : {r.get('grossProfit')}\n"
                        f"  Operating Income   : {r.get('operatingIncome')}\n"
                        f"  Net Income         : {r.get('netIncome')}\n"
                        f"  EPS (Diluted)      : {r.get('dilutedEPS')}\n"
                        f"  EBITDA             : {r.get('ebitda')}\n"
                        f"  R&D Expense        : {r.get('researchAndDevelopment')}\n"
                    )
                results.append("\n".join(block))
            except Exception as av_exc:
                results.append(f"[FinData] Income statement error: yfinance={yf_exc}, AV={av_exc}")

    # ── 3. Balance Sheet (yfinance primary, AV fallback) ─────────────────────
    if statement_type in ("balance", "all"):
        try:
            stock = yf.Ticker(ticker)
            bs = stock.balance_sheet if period == "annual" else stock.quarterly_balance_sheet
            block = [f"[FinData] Balance Sheet ({period.title()}) – {ticker}"]

            if bs is not None and not bs.empty:
                cols = bs.columns[:years]
                for col in cols:
                    date_str = col.strftime("%Y-%m-%d")
                    total_assets = bs.loc["Total Assets"].get(col) if "Total Assets" in bs.index else None
                    total_liab = bs.loc["Total Liabilities Net Minority Interest"].get(col) if "Total Liabilities Net Minority Interest" in bs.index else None
                    equity = bs.loc["Stockholders Equity"].get(col) if "Stockholders Equity" in bs.index else None
                    cash = bs.loc["Cash And Cash Equivalents"].get(col) if "Cash And Cash Equivalents" in bs.index else None
                    total_debt = bs.loc["Total Debt"].get(col) if "Total Debt" in bs.index else None
                    retained = bs.loc["Retained Earnings"].get(col) if "Retained Earnings" in bs.index else None
                    block.append(
                        f"\n  Period: {date_str}\n"
                        f"  Total Assets        : {int(total_assets) if total_assets else 'N/A'}\n"
                        f"  Total Liabilities   : {int(total_liab) if total_liab else 'N/A'}\n"
                        f"  Shareholder Equity  : {int(equity) if equity else 'N/A'}\n"
                        f"  Cash & Equivalents  : {int(cash) if cash else 'N/A'}\n"
                        f"  Total Debt          : {int(total_debt) if total_debt else 'N/A'}\n"
                        f"  Retained Earnings   : {int(retained) if retained else 'N/A'}\n"
                    )
                results.append("\n".join(block))
            else:
                raise ValueError("yfinance returned empty balance sheet")
        except Exception as yf_exc:
            # Fallback to AV
            try:
                data    = _av_fetch("BALANCE_SHEET", ticker)
                key     = "annualReports" if period == "annual" else "quarterlyReports"
                reports = data.get(key, [])[:years]
                block   = [f"[FinData] Balance Sheet ({period.title()}) – {ticker}"]
                if not reports:
                    block.append(f"\n  [Data unavailable] Neither yfinance ({yf_exc}) nor Alpha Vantage returned {period} balance sheet data.")
                for r in reports:
                    block.append(
                        f"\n  Period: {r.get('fiscalDateEnding')}\n"
                        f"  Total Assets        : {r.get('totalAssets')}\n"
                        f"  Total Liabilities   : {r.get('totalLiabilities')}\n"
                        f"  Shareholder Equity  : {r.get('totalShareholderEquity')}\n"
                        f"  Cash & Equivalents  : {r.get('cashAndCashEquivalentsAtCarryingValue')}\n"
                        f"  Total Debt          : {r.get('shortLongTermDebtTotal')}\n"
                        f"  Retained Earnings   : {r.get('retainedEarnings')}\n"
                    )
                results.append("\n".join(block))
            except Exception as av_exc:
                results.append(f"[FinData] Balance sheet error: yfinance={yf_exc}, AV={av_exc}")

    # ── 4. Cash Flow (yfinance primary, AV fallback) ─────────────────────────
    if statement_type in ("cashflow", "all"):
        try:
            stock = yf.Ticker(ticker)
            cf = stock.cashflow if period == "annual" else stock.quarterly_cashflow
            block = [f"[FinData] Cash Flow ({period.title()}) – {ticker}"]

            if cf is not None and not cf.empty:
                cols = cf.columns[:years]
                for col in cols:
                    date_str = col.strftime("%Y-%m-%d")
                    ocf = cf.loc["Operating Cash Flow"].get(col) if "Operating Cash Flow" in cf.index else None
                    capex = cf.loc["Capital Expenditure"].get(col) if "Capital Expenditure" in cf.index else None
                    fcf = cf.loc["Free Cash Flow"].get(col) if "Free Cash Flow" in cf.index else None
                    div = cf.loc["Cash Dividends Paid"].get(col) if "Cash Dividends Paid" in cf.index else None
                    buyback = cf.loc["Repurchase Of Capital Stock"].get(col) if "Repurchase Of Capital Stock" in cf.index else None
                    block.append(
                        f"\n  Period: {date_str}\n"
                        f"  Operating Cash Flow : {int(ocf) if ocf else 'N/A'}\n"
                        f"  CapEx               : {int(capex) if capex else 'N/A'}\n"
                        f"  Free Cash Flow      : {int(fcf) if fcf else 'N/A'}\n"
                        f"  Dividends Paid      : {int(div) if div else 'N/A'}\n"
                        f"  Share Buybacks      : {int(buyback) if buyback else 'N/A'}\n"
                    )
                results.append("\n".join(block))
            else:
                raise ValueError("yfinance returned empty cash flow")
        except Exception as yf_exc:
            try:
                data    = _av_fetch("CASH_FLOW", ticker)
                key     = "annualReports" if period == "annual" else "quarterlyReports"
                reports = data.get(key, [])[:years]
                block   = [f"[FinData] Cash Flow ({period.title()}) – {ticker}"]
                if not reports:
                    block.append(f"\n  [Data unavailable] Neither yfinance ({yf_exc}) nor Alpha Vantage returned {period} cash flow data.")
                for r in reports:
                    block.append(
                        f"\n  Period: {r.get('fiscalDateEnding')}\n"
                        f"  Operating Cash Flow : {r.get('operatingCashflow')}\n"
                        f"  CapEx               : {r.get('capitalExpenditures')}\n"
                        f"  Dividends Paid      : {r.get('dividendPayout')}\n"
                        f"  Share Buybacks      : {r.get('repurchaseOfCommonStock')}\n"
                    )
                results.append("\n".join(block))
            except Exception as av_exc:
                results.append(f"[FinData] Cash flow error: yfinance={yf_exc}, AV={av_exc}")

    # ── 5. Key Ratios / Company Overview (Alpha Vantage) ─────────────────────
    if statement_type in ("ratios", "all"):
        try:
            data  = _av_fetch("OVERVIEW", ticker)
            ratios = {
                "PE Ratio":          data.get("PERatio"),
                "PEG Ratio":         data.get("PEGRatio"),
                "Book Value":        data.get("BookValue"),
                "EPS":               data.get("EPS"),
                "ROE (%)":           data.get("ReturnOnEquityTTM"),
                "ROA (%)":           data.get("ReturnOnAssetsTTM"),
                "Operating Margin":  data.get("OperatingMarginTTM"),
                "Profit Margin":     data.get("ProfitMargin"),
                "Quarterly Revenue Growth": data.get("QuarterlyRevenueGrowthYOY"),
                "Quarterly Earnings Growth": data.get("QuarterlyEarningsGrowthYOY"),
                "Analyst Target Price":    data.get("AnalystTargetPrice"),
                "52-Week High":      data.get("52WeekHigh"),
                "52-Week Low":       data.get("52WeekLow"),
            }
            # Check if ALL values are null — means AV returned an empty shell
            if all(v is None for v in ratios.values()):
                results.append(
                    f"[FinData] Key Ratios – {ticker}\n"
                    f"{'='*60}\n"
                    f"[Data unavailable] Alpha Vantage returned no ratio data for {ticker}. "
                    f"This may be due to API rate limits or an invalid/missing ALPHA_VANTAGE_KEY."
                )
            else:
                results.append(
                    f"[FinData] Key Ratios – {ticker}\n"
                    f"{'='*60}\n"
                    f"{json.dumps(ratios, indent=2)}"
                )
        except Exception as exc:
            results.append(f"[FinData] Ratios error: {exc}")

    return "\n\n".join(results) if results else f"[FinData] No data retrieved for {ticker}."
