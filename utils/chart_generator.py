"""
utils/chart_generator.py — Chart library for financial reports.

The PLANNER specifies which charts to produce by including a "charts" list
in report_outline. Each entry: {"chart_type": "revenue_trend", "section": "EARNINGS_ANALYSIS"}.
The report_generator calls generate_chart_for_section() which looks up the
chart_type in CHART_REGISTRY and runs it if the data exists.

If the planner didn't specify charts (backward compat), a default mapping
from section_id → chart_type is used as fallback.
"""
from __future__ import annotations
import os
import re
import json
from typing import Optional, Callable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f8f9fa",
    "axes.edgecolor":   "#dee2e6",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
    "font.size":        10,
    "figure.dpi":       150,
})
COLORS = ["#0d6efd", "#198754", "#ffc107", "#dc3545", "#6f42c1", "#20c997", "#fd7e14", "#6610f2"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Data extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_market_overview(chunks: list[dict]) -> Optional[dict]:
    for c in chunks:
        if c.get("status") != "success":
            continue
        content = str(c.get("content", ""))
        if "[FinData] Market Overview" not in content:
            continue
        match = re.search(r"\{[^}]+\"Company Name\"[^}]+\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return None


def _parse_income_statements(chunks: list[dict]) -> list[dict]:
    periods = []
    for c in chunks:
        if c.get("status") != "success":
            continue
        content = str(c.get("content", ""))
        for block in re.finditer(
            r"Period:\s*(\d{4}-\d{2}-\d{2})\s*\n\s*Total Revenue\s*:\s*(\d+)", content
        ):
            year = block.group(1)[:4]
            revenue = int(block.group(2))
            block_text = content[block.start():block.start()+600]
            net_match = re.search(r"Net Income\s*:\s*(\d+)", block_text)
            gross_match = re.search(r"Gross Profit\s*:\s*(\d+)", block_text)
            op_match = re.search(r"Operating Income\s*:\s*(\d+)", block_text)
            ebitda_match = re.search(r"EBITDA\s*:\s*(\d+)", block_text)
            rd_match = re.search(r"R&D Expense\s*:\s*(\d+)", block_text)
            capex_match = re.search(r"CapEx\s*:\s*(\d+)", block_text)
            ocf_match = re.search(r"Operating Cash Flow\s*:\s*(\d+)", block_text)
            periods.append({
                "year": year, "revenue": revenue,
                "net_income": int(net_match.group(1)) if net_match else None,
                "gross_profit": int(gross_match.group(1)) if gross_match else None,
                "operating_income": int(op_match.group(1)) if op_match else None,
                "ebitda": int(ebitda_match.group(1)) if ebitda_match else None,
                "rd_expense": int(rd_match.group(1)) if rd_match else None,
                "capex": int(capex_match.group(1)) if capex_match else None,
                "ocf": int(ocf_match.group(1)) if ocf_match else None,
            })
    seen = set()
    unique = []
    for p in sorted(periods, key=lambda x: x["year"]):
        if p["year"] not in seen:
            seen.add(p["year"])
            unique.append(p)
    return unique


def _parse_balance_sheet(chunks: list[dict]) -> list[dict]:
    periods = []
    for c in chunks:
        if c.get("status") != "success":
            continue
        content = str(c.get("content", ""))
        for block in re.finditer(
            r"Period:\s*(\d{4}-\d{2}-\d{2})\s*\n\s*Total Assets\s*:\s*(\d+)", content
        ):
            year = block.group(1)[:4]
            block_text = content[block.start():block.start()+500]
            debt_match = re.search(r"Total Debt\s*:\s*(\d+)", block_text)
            equity_match = re.search(r"Shareholder Equity\s*:\s*(\d+)", block_text)
            cash_match = re.search(r"Cash & Equivalents\s*:\s*(\d+)", block_text)
            liab_match = re.search(r"Total Liabilities\s*:\s*(\d+)", block_text)
            periods.append({
                "year": year,
                "total_assets": int(block.group(2)),
                "total_debt": int(debt_match.group(1)) if debt_match else None,
                "equity": int(equity_match.group(1)) if equity_match else None,
                "cash": int(cash_match.group(1)) if cash_match else None,
                "liabilities": int(liab_match.group(1)) if liab_match else None,
            })
    seen = set()
    unique = []
    for p in sorted(periods, key=lambda x: x["year"]):
        if p["year"] not in seen:
            seen.add(p["year"])
            unique.append(p)
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
#  Chart functions — each returns filepath or None
# ═══════════════════════════════════════════════════════════════════════════════

def chart_revenue_trend(chunks, output_dir, ticker) -> Optional[str]:
    """Bar chart: revenue + net income over available years."""
    periods = _parse_income_statements(chunks)
    if len(periods) < 2:
        return None
    years = [p["year"] for p in periods]
    revenues = [p["revenue"] / 1e9 for p in periods]
    net_incomes = [(p["net_income"] or 0) / 1e9 for p in periods]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(years))
    w = 0.35
    b1 = ax.bar([i - w/2 for i in x], revenues, w, label="Revenue", color=COLORS[0])
    b2 = ax.bar([i + w/2 for i in x], net_incomes, w, label="Net Income", color=COLORS[1])
    ax.set_xlabel("Fiscal Year"); ax.set_ylabel("USD (Billions)")
    ax.set_title(f"{ticker} — Revenue & Net Income Trend")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"FY{y}" for y in years]); ax.legend()
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"${bar.get_height():.0f}B", ha="center", fontsize=8)
    for bar in b2:
        if bar.get_height() > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"${bar.get_height():.0f}B", ha="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_revenue_trend.png")
    fig.savefig(path); plt.close(fig)
    return path


def chart_margin_trend(chunks, output_dir, ticker) -> Optional[str]:
    """Line chart: gross/operating/net margins over years."""
    periods = _parse_income_statements(chunks)
    if len(periods) < 2:
        return None
    years = [p["year"] for p in periods]
    has_data = False
    gross_m, op_m, net_m = [], [], []
    for p in periods:
        rev = p["revenue"]
        if rev and rev > 0:
            gm = (p["gross_profit"] / rev * 100) if p["gross_profit"] else None
            om = (p["operating_income"] / rev * 100) if p["operating_income"] else None
            nm = (p["net_income"] / rev * 100) if p["net_income"] else None
            gross_m.append(gm); op_m.append(om); net_m.append(nm)
            if any([gm, om, nm]): has_data = True
        else:
            gross_m.append(None); op_m.append(None); net_m.append(None)
    if not has_data:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(years))
    if any(v is not None for v in gross_m):
        ax.plot(list(x), [v or 0 for v in gross_m], "o-", label="Gross Margin", color=COLORS[0], linewidth=2)
    if any(v is not None for v in op_m):
        ax.plot(list(x), [v or 0 for v in op_m], "s-", label="Operating Margin", color=COLORS[1], linewidth=2)
    if any(v is not None for v in net_m):
        ax.plot(list(x), [v or 0 for v in net_m], "^-", label="Net Margin", color=COLORS[3], linewidth=2)
    ax.set_xlabel("Fiscal Year"); ax.set_ylabel("Margin (%)")
    ax.set_title(f"{ticker} — Profitability Margins")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"FY{y}" for y in years]); ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_margins.png")
    fig.savefig(path); plt.close(fig)
    return path


def chart_valuation_multiples(chunks, output_dir, ticker) -> Optional[str]:
    """Horizontal bar: P/E, Forward P/E, EV/EBITDA, P/B."""
    overview = _parse_market_overview(chunks)
    if not overview:
        return None
    metrics = {}
    for key, label in [("Trailing P/E", "Trailing P/E"), ("Forward P/E", "Forward P/E"),
                        ("EV/EBITDA", "EV/EBITDA"), ("Price/Book", "Price/Book")]:
        val = overview.get(key)
        if val is not None and val != 0:
            try: metrics[label] = float(val)
            except (ValueError, TypeError): pass
    if len(metrics) < 2:
        return None
    labels = list(metrics.keys()); values = list(metrics.values())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, values, color=COLORS[:len(labels)], height=0.5)
    ax.set_xlabel("Multiple"); ax.set_title(f"{ticker} — Valuation Multiples")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f"{val:.1f}×", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_valuation_multiples.png")
    fig.savefig(path); plt.close(fig)
    return path


def chart_price_vs_ma(chunks, output_dir, ticker) -> Optional[str]:
    """Bar: current price vs 50-day MA, 200-day MA, 52W range."""
    overview = _parse_market_overview(chunks)
    if not overview:
        return None
    price = overview.get("Current Price"); ma50 = overview.get("50-Day MA")
    ma200 = overview.get("200-Day MA"); high52 = overview.get("52-Week High"); low52 = overview.get("52-Week Low")
    if not all([price, ma50, ma200]):
        return None
    labels = ["52W Low", "200-Day MA", "50-Day MA", "Current", "52W High"]
    values = [float(low52 or 0), float(ma200), float(ma50), float(price), float(high52 or 0)]
    colors_bar = ["#dc3545", "#6c757d", "#0d6efd", "#198754", "#ffc107"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(labels, values, color=colors_bar, height=0.5)
    ax.set_xlabel("Price (USD)"); ax.set_title(f"{ticker} — Price Position")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f"${val:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_price_position.png")
    fig.savefig(path); plt.close(fig)
    return path


def chart_debt_vs_equity(chunks, output_dir, ticker) -> Optional[str]:
    """Stacked bar: debt vs equity over years."""
    bs = _parse_balance_sheet(chunks)
    if len(bs) < 2:
        return None
    years = [p["year"] for p in bs]
    debts = [(p["total_debt"] or 0) / 1e9 for p in bs]
    equities = [(p["equity"] or 0) / 1e9 for p in bs]
    if not any(d > 0 for d in debts) and not any(e > 0 for e in equities):
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(years))
    ax.bar(x, equities, label="Shareholder Equity", color=COLORS[1])
    ax.bar(x, debts, bottom=equities, label="Total Debt", color=COLORS[3])
    ax.set_xlabel("Fiscal Year"); ax.set_ylabel("USD (Billions)")
    ax.set_title(f"{ticker} — Debt vs Equity"); ax.set_xticks(list(x))
    ax.set_xticklabels([f"FY{y}" for y in years]); ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_debt_equity.png")
    fig.savefig(path); plt.close(fig)
    return path


def chart_cash_flow(chunks, output_dir, ticker) -> Optional[str]:
    """Bar chart: operating cash flow vs capex over years."""
    periods = _parse_income_statements(chunks)
    if len(periods) < 2:
        return None
    has_cf = any(p.get("ocf") for p in periods)
    if not has_cf:
        return None
    years = [p["year"] for p in periods]
    ocfs = [(p.get("ocf") or 0) / 1e9 for p in periods]
    capexs = [(p.get("capex") or 0) / 1e9 for p in periods]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(years))
    w = 0.35
    ax.bar([i-w/2 for i in x], ocfs, w, label="Operating Cash Flow", color=COLORS[0])
    ax.bar([i+w/2 for i in x], capexs, w, label="CapEx", color=COLORS[3])
    ax.set_xlabel("Fiscal Year"); ax.set_ylabel("USD (Billions)")
    ax.set_title(f"{ticker} — Cash Flow vs CapEx")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"FY{y}" for y in years]); ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_cashflow.png")
    fig.savefig(path); plt.close(fig)
    return path


def chart_rd_spend(chunks, output_dir, ticker) -> Optional[str]:
    """Bar chart: R&D spending trend."""
    periods = _parse_income_statements(chunks)
    if len(periods) < 2 or not any(p.get("rd_expense") for p in periods):
        return None
    years = [p["year"] for p in periods]
    rds = [(p.get("rd_expense") or 0) / 1e9 for p in periods]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(years)), rds, color=COLORS[4])
    ax.set_xlabel("Fiscal Year"); ax.set_ylabel("USD (Billions)")
    ax.set_title(f"{ticker} — R&D Spending")
    ax.set_xticks(range(len(years))); ax.set_xticklabels([f"FY{y}" for y in years])
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"${bar.get_height():.1f}B", ha="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_rd_spend.png")
    fig.savefig(path); plt.close(fig)
    return path


def chart_ebitda_trend(chunks, output_dir, ticker) -> Optional[str]:
    """Bar chart: EBITDA trend."""
    periods = _parse_income_statements(chunks)
    if len(periods) < 2 or not any(p.get("ebitda") for p in periods):
        return None
    years = [p["year"] for p in periods]
    ebitdas = [(p.get("ebitda") or 0) / 1e9 for p in periods]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(years)), ebitdas, color=COLORS[5])
    ax.set_xlabel("Fiscal Year"); ax.set_ylabel("USD (Billions)")
    ax.set_title(f"{ticker} — EBITDA Trend")
    ax.set_xticks(range(len(years))); ax.set_xticklabels([f"FY{y}" for y in years])
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"${bar.get_height():.0f}B", ha="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{ticker}_ebitda.png")
    fig.savefig(path); plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Registry + planner-driven selection
# ═══════════════════════════════════════════════════════════════════════════════

# Chart type name → function. The planner uses these names in its output.
CHART_REGISTRY: dict[str, Callable] = {
    "revenue_trend":        chart_revenue_trend,
    "margin_trend":         chart_margin_trend,
    "valuation_multiples":  chart_valuation_multiples,
    "price_position":       chart_price_vs_ma,
    "debt_vs_equity":       chart_debt_vs_equity,
    "cash_flow":            chart_cash_flow,
    "rd_spend":             chart_rd_spend,
    "ebitda_trend":         chart_ebitda_trend,
}

# Fallback: if planner didn't specify charts, use this section → chart mapping
DEFAULT_SECTION_CHARTS: dict[str, list[str]] = {
    "EXEC_SUMMARY":          ["revenue_trend"],
    "EARNINGS_ANALYSIS":     ["revenue_trend", "margin_trend"],
    "VALUATION":             ["valuation_multiples"],
    "INVESTMENT_THESIS":     ["price_position"],
    "RISK_ASSESSMENT":       ["debt_vs_equity"],
    "GROWTH_DRIVERS":        ["rd_spend", "ebitda_trend"],
    "COMPANY_PROFILE":       ["price_position"],
    "FINANCIAL_COMPARISON":  ["revenue_trend", "margin_trend"],
    "VALUATION_COMPARISON":  ["valuation_multiples"],
    "VALUATION_MODELS":      ["valuation_multiples"],
    "EVENT_SUMMARY":         ["revenue_trend"],
    "MARKET_REACTION":       ["price_position"],
    "APPENDIX":              ["cash_flow"],
}

# Names for the planner prompt so it knows what chart types exist
CHART_TYPE_DESCRIPTIONS = """
Available chart types (use these exact names in the "charts" field):
  revenue_trend       — Bar chart: revenue + net income over years
  margin_trend        — Line chart: gross/operating/net margins over years
  valuation_multiples — Horizontal bar: P/E, Forward P/E, EV/EBITDA, P/B
  price_position      — Bar: current price vs 50-day MA, 200-day MA, 52W range
  debt_vs_equity      — Stacked bar: debt vs equity over years
  cash_flow           — Bar: operating cash flow vs capex over years
  rd_spend            — Bar: R&D spending trend over years
  ebitda_trend        — Bar: EBITDA trend over years
"""


def generate_charts_for_section(section_id: str, chunks: list[dict],
                                 output_dir: str, ticker: str,
                                 planned_charts: list[dict] = None) -> list[str]:
    """Generate charts for a section. Uses planner's chart specs if provided,
    falls back to DEFAULT_SECTION_CHARTS mapping otherwise."""
    os.makedirs(output_dir, exist_ok=True)

    # Determine which chart types to generate for this section
    if planned_charts:
        chart_types = [
            c.get("chart_type") for c in planned_charts
            if c.get("section") == section_id and c.get("chart_type") in CHART_REGISTRY
        ]
    else:
        chart_types = DEFAULT_SECTION_CHARTS.get(section_id, [])

    paths = []
    for ct in chart_types:
        fn = CHART_REGISTRY.get(ct)
        if fn:
            try:
                path = fn(chunks, output_dir, ticker)
                if path:
                    paths.append(path)
            except Exception:
                pass
    return paths


def generate_all_charts(chunks: list[dict], output_dir: str, ticker: str,
                        report_type: str = "COMPANY_RESEARCH") -> list[str]:
    """Fallback: generate all applicable charts."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, fn in CHART_REGISTRY.items():
        try:
            path = fn(chunks, output_dir, ticker)
            if path:
                paths.append(path)
                print(f"  [Charts] Generated: {os.path.basename(path)}")
        except Exception as exc:
            print(f"  [Charts] Skipped {name}: {exc}")
    return paths
