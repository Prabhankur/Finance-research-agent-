"""
tools/sec_filing.py  –  SEC EDGAR filing retrieval using edgartools + data.sec.gov XBRL API.

TWO complementary data sources:
  1. edgartools (pip install edgartools)
     → Fetches and parses 10-K/10-Q/8-K into clean text sections
     → Extracts: Item 1 (Business), Item 1A (Risk Factors),
                 Item 7 (MD&A), Item 7A (Market Risk)
     → Free, no API key, MIT licensed

  2. data.sec.gov/api/xbrl/companyfacts
     → Structured XBRL financial numbers (revenue, net income, EPS etc.)
     → Free, no API key, official SEC REST API

Reference:
  https://github.com/dgunning/edgartools
  https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

from __future__ import annotations
import re
import time
import requests
from typing import Optional

_HEADERS = {
    "User-Agent": "FinancialReportAgent/1.0 (developer@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Key 10-K sections we want for the report
_SECTIONS_10K = {
    "1":  "Business Overview",
    "1A": "Risk Factors",
    "7":  "Management Discussion & Analysis (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
}

# Key XBRL concepts we pull from companyfacts API.
#
# Revenue maps to MULTIPLE tags, tried in order, not just one. This is a real
# production bug fix: "Revenues" is the pre-ASC-606 tag. Most large filers
# (Apple included) switched to RevenueFromContractWithCustomerExcluding
# AssessedTax around FY2018 when they adopted the new revenue-recognition
# standard, and simply STOPPED reporting facts under "Revenues" afterward.
# Requesting FY2024 revenue under only the old tag found zero FY2024 facts,
# and the display logic below used to silently fall back to the newest fact
# it COULD find under that tag — which was ~FY2018 — and showed it as if it
# were the requested year, with no indication the period had silently
# changed. ($265.6B is almost exactly Apple's real FY2018 revenue.)
_XBRL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "Total Revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",  # pre-ASC-606 fallback, for older filings only
    ),
    "Net Income":            ("NetIncomeLoss",),
    "EPS (Diluted)":         ("EarningsPerShareDiluted",),
    "Gross Profit":          ("GrossProfit",),
    "Operating Income":      ("OperatingIncomeLoss",),
    "R&D Expense":           ("ResearchAndDevelopmentExpense",),
    "Cash & Equivalents":    ("CashAndCashEquivalentsAtCarryingValue",),
    "Long-Term Debt":        ("LongTermDebt",),
    "Shareholders Equity":   ("StockholdersEquity",),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: resolve ticker → CIK via SEC mapping
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_cik(ticker: str) -> Optional[str]:
    """Returns zero-padded 10-digit CIK, or None."""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
        for _, entry in resp.json().items():
            if entry.get("ticker", "").upper() == clean_ticker:
                return str(entry["cik_str"]).zfill(10)
        return None
    except Exception as exc:
        print(f"[SEC] CIK lookup error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Source 1: edgartools – clean section text from 10-K/10-Q
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_via_edgartools(ticker: str, filing_type: str, year: Optional[int]) -> str:
    """
    Uses edgartools to fetch and extract clean text sections from a filing.
    Returns formatted multi-section text block.
    """
    try:
        from edgar import Company
        from edgar.core import set_identity

        # SEC requires identity header
        set_identity("FinancialReportAgent developer@example.com")

        company  = Company(ticker)
        filings  = company.get_filings(form=filing_type)

        if not filings:
            return f"[SEC] No {filing_type} filings found for {ticker} via edgartools."

        # Filter by year if requested
        if year:
            year_filings = [
                f for f in filings
                if hasattr(f, "filing_date") and str(year) in str(f.filing_date)
            ]
            target_filing = year_filings[0] if year_filings else filings[0]
        else:
            target_filing = filings[0]

        filing_date = getattr(target_filing, "filing_date", "Unknown")
        print(f"[SEC] edgartools found {filing_type} filed {filing_date} for {ticker}")

        # ── Get the actual filing document ────────────────────────────────────
        doc = target_filing.obj()

        output_lines = [
            f"[SEC] {filing_type} for {ticker}",
            f"Filing Date : {filing_date}",
            "=" * 60,
        ]

        if filing_type == "10-K":
            # Extract key sections
            sections_extracted = 0
            for section_id, section_name in _SECTIONS_10K.items():
                try:
                    section_text = doc[section_id]
                    if section_text:
                        # Convert to string cleanly
                        if hasattr(section_text, "text"):
                            clean = section_text.text
                        else:
                            clean = str(section_text)

                        # Strip residual HTML if any slipped through
                        clean = re.sub(r"<[^>]{1,200}>", " ", clean)
                        clean = re.sub(r"\s{3,}", "\n", clean).strip()

                        # Cap each section at 2500 chars
                        preview = clean[:2_500]
                        output_lines.append(
                            f"\n--- Item {section_id}: {section_name} ---\n{preview}"
                            + ("\n[...truncated]" if len(clean) > 2_500 else "")
                        )
                        sections_extracted += 1
                except Exception as sec_err:
                    output_lines.append(f"\n--- Item {section_id}: {section_name} ---\n[Not available: {sec_err}]")

            if sections_extracted == 0:
                # Fallback: get markdown representation of entire filing
                try:
                    md = doc.markdown() if hasattr(doc, "markdown") else str(doc)
                    output_lines.append(md[:6_000] + "\n[...truncated]")
                except Exception:
                    output_lines.append("[SEC] Could not extract section text from this filing.")

        elif filing_type in ("10-Q", "8-K"):
            try:
                # For 10-Q and 8-K use markdown/text representation
                if hasattr(doc, "markdown"):
                    content = doc.markdown()[:6_000]
                elif hasattr(doc, "text"):
                    content = doc.text[:6_000]
                else:
                    content = str(doc)[:6_000]
                output_lines.append(content + "\n[...truncated]")
            except Exception as e:
                output_lines.append(f"[SEC] Could not extract {filing_type} content: {e}")

        else:
            # DEF 14A and others
            try:
                content = str(doc)[:5_000]
                output_lines.append(content + "\n[...truncated]")
            except Exception:
                output_lines.append("[SEC] Could not extract content from this filing type.")

        return "\n".join(output_lines)

    except ImportError:
        return "[SEC] edgartools not installed. Run: pip install edgartools"
    except Exception as exc:
        return f"[SEC] edgartools error for {ticker}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
#  Source 2: data.sec.gov XBRL companyfacts – structured financial numbers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_xbrl_facts(cik: str, ticker: str, year: Optional[int], filing_type: str = "10-K") -> str:
    """
    Pulls key financial metrics from the SEC XBRL Company Facts API.
    Returns a formatted table of figures.

    filing_type controls which period type we pull: "10-K" filters to annual
    (fp == "FY") facts, "10-Q" filters to quarterly (fp in Q1/Q2/Q3) facts.
    Previously this only ever pulled annual data regardless of what was
    asked for, so a 10-Q request got no structured numbers at all — this
    was the actual data gap behind the original bug report, separate from
    the edgartools install issue.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    is_quarterly = filing_type == "10-Q"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        facts = resp.json()

        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            return "[SEC XBRL] No us-gaap facts found."

        period_label = "Quarterly" if is_quarterly else "Annual"
        lines = [
            f"\n[SEC XBRL] Structured {period_label} Financial Data for {ticker}",
            "=" * 60,
        ]

        for label, concepts in _XBRL_CONCEPTS.items():
            # Merge facts across every alternate tag for this line item (e.g.
            # revenue's pre- and post-ASC-606 tags) instead of stopping at
            # the first tag that exists at all — a company can have SOME
            # facts under an old tag and MORE RECENT facts only under a new
            # one, and only trying the first would silently miss the newer
            # data. See _XBRL_CONCEPTS' comment for the real bug this fixes.
            usd_facts: list = []
            for concept in concepts:
                concept_data = us_gaap.get(concept, {})
                usd_facts.extend(concept_data.get("units", {}).get("USD", []))
                if not concept_data.get("units", {}).get("USD"):
                    # Try shares unit for EPS
                    usd_facts.extend(concept_data.get("units", {}).get("USD/shares", []))

            if is_quarterly:
                period_facts = [
                    f for f in usd_facts
                    if f.get("form") in ("10-Q",) and f.get("fp") in ("Q1", "Q2", "Q3")
                ]
            else:
                period_facts = [
                    f for f in usd_facts
                    if f.get("form") in ("10-K", "20-F", "40-F") and f.get("fp") == "FY"
                ]

            if not period_facts:
                lines.append(f"  {label:45s}: N/A")
                continue

            # Sort by end date descending
            period_facts.sort(key=lambda x: x.get("end", ""), reverse=True)

            # Filter by year if requested
            if year:
                yr_data = [f for f in period_facts if str(year) in f.get("end", "")]
                if yr_data:
                    display = yr_data[0]
                else:
                    # NEVER silently substitute a different year's figure —
                    # this is the exact production bug: FY2018 revenue got
                    # shown as if it were the requested FY2024 figure, with
                    # nothing indicating the period had changed. Say plainly
                    # that the requested year isn't available under any known
                    # tag, and name what actually IS available so the caller
                    # (the LLM, and ultimately the user) sees the honest gap
                    # instead of a silently wrong number.
                    latest = period_facts[0]
                    lines.append(
                        f"  {label:45s}: N/A for FY{year} "
                        f"(most recent available: {latest.get('end', '?')})"
                    )
                    continue
            else:
                display = period_facts[0]

            val      = display.get("val", 0)
            end_date = display.get("end", "?")

            # Format large numbers
            if abs(val) >= 1_000_000_000:
                formatted = f"${val/1_000_000_000:.2f}B"
            elif abs(val) >= 1_000_000:
                formatted = f"${val/1_000_000:.2f}M"
            elif abs(val) >= 1_000:
                formatted = f"${val/1_000:.2f}K"
            else:
                formatted = f"{val:.4f}"

            period_end_label = "Q end" if is_quarterly else "FY end"
            lines.append(f"  {label:45s}: {formatted:>12}  ({period_end_label}: {end_date})")

        return "\n".join(lines)

    except requests.exceptions.HTTPError as exc:
        return f"[SEC XBRL] HTTP error {exc.response.status_code}: CIK {cik} may not have XBRL data."
    except Exception as exc:
        return f"[SEC XBRL] Error fetching company facts: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
#  Main tool function
# ─────────────────────────────────────────────────────────────────────────────

def sec_filing_search(
    ticker: str,
    filing_type: str = "10-K",
    year: Optional[int] = None,
) -> str:
    """
    Retrieves SEC filings for a given company.

    For 10-K filings returns BOTH:
      • Clean section text (Business, Risk Factors, MD&A) via edgartools
      • Structured XBRL financial numbers via data.sec.gov

    For 10-Q / 8-K returns clean text via edgartools.

    Args:
        ticker:       Stock ticker, e.g. "AAPL", "MSFT".
                      Non-US tickers (TCS.NS) return a clear message.
        filing_type:  One of "10-K", "10-Q", "8-K", "DEF 14A".
        year:         Optional calendar year filter, e.g. 2024.

    Returns:
        Rich formatted string: section text + XBRL numbers, ready for LLM context.
    """
    ticker = ticker.upper().strip()
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

    print(f"[SEC] Fetching {filing_type} for {ticker}" + (f" ({year})" if year else "") + "...")
    time.sleep(0.2)   # polite delay per SEC fair-use policy

    # ── Check if this is a non-US ticker ─────────────────────────────────────
    if ticker != clean_ticker:
        return (
            f"[SEC] '{ticker}' appears to be a non-US ticker. "
            f"SEC EDGAR only covers US-listed companies. "
            f"For {clean_ticker}, use financial_data_api and web_search instead."
        )

    # ── Source 1: Clean section text via edgartools ───────────────────────────
    section_text = _fetch_via_edgartools(clean_ticker, filing_type, year)

    # ── Source 2: XBRL numbers (10-K annual, or 10-Q quarterly) ───────────────
    xbrl_text = ""
    if filing_type in ("10-K", "10-Q"):
        cik = _resolve_cik(clean_ticker)
        if cik:
            time.sleep(0.2)
            xbrl_text = _fetch_xbrl_facts(cik, clean_ticker, year, filing_type=filing_type)
        else:
            xbrl_text = f"[SEC XBRL] Could not resolve CIK for {clean_ticker}."

    return "\n\n".join(filter(None, [section_text, xbrl_text]))