"""
tools/fred.py – Federal Reserve Economic Data (FRED) API wrapper.

Provides access to US and international macro time-series: GDP, CPI/PCE
inflation, Fed funds rate, unemployment, industrial production, and many
others. Free API — requires a FRED_API_KEY env var (register at
https://fred.stlouisfed.org/docs/api/api_key.html).

Common series shortcuts (accepted as `series_id` values):
  gdp             → GDP (billions of dollars, quarterly)
  cpi             → CPIAUCSL (Consumer Price Index)
  core_cpi        → CPILFESL (Core CPI, ex food & energy)
  fed_funds       → DFF (effective federal funds rate)
  unemployment    → UNRATE
  10y_yield       → DGS10 (10-year Treasury yield)
  2y_yield        → DGS2
  yield_curve     → T10Y2Y (10y minus 2y spread)
  industrial_prod → INDPRO
  m2              → M2SL
  pce_inflation   → PCEPILFE (Core PCE, Fed's preferred inflation gauge)

Or pass any raw FRED series ID directly (e.g. "CLVMNACSCAB1GQEU272020" for
Eurozone GDP).
"""
from __future__ import annotations
import os
import requests
from typing import Optional
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

_SHORTCUTS = {
    "gdp":              "GDP",
    "cpi":              "CPIAUCSL",
    "core_cpi":         "CPILFESL",
    "fed_funds":        "DFF",
    "unemployment":     "UNRATE",
    "10y_yield":        "DGS10",
    "2y_yield":         "DGS2",
    "yield_curve":      "T10Y2Y",
    "industrial_prod":  "INDPRO",
    "m2":               "M2SL",
    "pce_inflation":    "PCEPILFE",
    "retail_sales":     "RSAFS",
    "housing_starts":   "HOUST",
    "consumer_sent":    "UMCSENT",
}


_META_URL = "https://api.stlouisfed.org/fred/series"


def resolve_series_id(series_id: str) -> str:
    """Shortcut ('gdp') -> real FRED ID ('GDP'), same rule _fred_fetch uses.

    Exposed so the storage layer keys macro_series by the REAL id. Before
    this, a call made with the shortcut 'gdp' would have registered a second
    series row 'gdp' next to 'GDP' for the very same data.
    """
    code = (series_id or "").strip()          # ' gdp ' must resolve like 'gdp'
    return _SHORTCUTS.get(code.lower(), code.upper())


def _fetch_series_metadata(resolved: str, api_key: str) -> Optional[dict]:
    """
    GET /fred/series — the endpoint that carries the title, units, frequency
    and seasonal-adjustment of a series. /series/observations (the only one
    this tool used to call) returns dates and values and NOTHING describing
    them, which is why the agent could not tell a Seasonally Adjusted Annual
    Rate series from a raw quarterly one.

    Strictly best-effort: any failure returns None and the tool behaves
    exactly as it did before. Metadata is an enrichment, never a dependency.
    """
    try:
        resp = requests.get(
            _META_URL,
            params={"series_id": resolved, "api_key": api_key, "file_type": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        seriess = resp.json().get("seriess") or []
        if not seriess:
            return None
        row = seriess[0]
        return {
            "title":      row.get("title"),
            "units":      row.get("units"),
            "frequency":  row.get("frequency"),
            "adjustment": row.get("seasonal_adjustment"),
        }
    except Exception:
        return None


def _fred_fetch(series_id: str, observation_start: Optional[str] = None,
                observation_end: Optional[str] = None,
                limit: int = 60) -> str:
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        return "[FRED] Error: FRED_API_KEY environment variable is missing. Register free at https://fred.stlouisfed.org/docs/api/api_key.html"

    # Resolve shortcut to real series ID
    resolved = _SHORTCUTS.get(series_id.lower(), series_id.upper())

    params = {
        "series_id":         resolved,
        "api_key":           api_key,
        "file_type":         "json",
        "sort_order":        "desc",
        "limit":             limit,
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end

    try:
        resp = requests.get(_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code
        # 400 usually means the series ID doesn't exist
        if code == 400:
            return f"[FRED] Series '{resolved}' not found. Check the series ID at https://fred.stlouisfed.org/searchresults?st={resolved}"
        return f"[FRED] HTTP error {code}: {exc}"
    except requests.exceptions.Timeout:
        return "[FRED] Request timed out."
    except Exception as exc:
        return f"[FRED] Error fetching series: {exc}"

    observations = data.get("observations", [])
    if not observations:
        return f"[FRED] No observations returned for series '{resolved}'."

    # Format as a compact readable table
    meta = _fetch_series_metadata(resolved, api_key)
    lines = [
        f"[FRED] Series: {resolved}  ({series_id if series_id != resolved else 'raw series ID'})",
    ]
    if meta:
        if meta.get("title"):
            lines.append(f"Title: {meta['title']}")
        lines.append(
            f"Units: {meta.get('units') or 'n/a'} | "
            f"Frequency: {meta.get('frequency') or 'n/a'} | "
            f"Adjustment: {meta.get('adjustment') or 'n/a'}"
        )
    lines += [
        "=" * 60,
        f"{'Date':<12}  {'Value':>15}",
        "-" * 30,
    ]
    for obs in observations[:limit]:
        d = obs.get("date", "?")
        v = obs.get("value", ".")
        # FRED uses "." for missing values
        if v == ".":
            lines.append(f"{d:<12}  {'N/A':>15}")
        else:
            try:
                # Format numbers nicely
                num = float(v)
                if abs(num) >= 1000:
                    formatted = f"{num:,.2f}"
                else:
                    formatted = f"{num:.4f}".rstrip("0").rstrip(".")
                lines.append(f"{d:<12}  {formatted:>15}")
            except ValueError:
                lines.append(f"{d:<12}  {v:>15}")

    # Add a small trend note if we have at least 2 numeric points
    numeric_vals = [(o["date"], float(o["value"])) for o in observations
                     if o.get("value", ".") != "."]
    if len(numeric_vals) >= 2:
        latest_date, latest_val = numeric_vals[0]
        older_date, older_val = numeric_vals[-1]
        if older_val != 0:
            pct = (latest_val - older_val) / abs(older_val) * 100
            lines.append("")
            lines.append(f"Change over range: {older_val:,.4f} ({older_date}) → {latest_val:,.4f} ({latest_date})  ≈ {pct:+.2f}%")

    return "\n".join(lines)


def fred_series(series_id: str,
                 observation_start: Optional[str] = None,
                 observation_end: Optional[str] = None,
                 limit: int = 60) -> str:
    """
    Fetches a macroeconomic time-series from FRED.

    Args:
        series_id: Either a shortcut (gdp, cpi, fed_funds, unemployment, 10y_yield,
                   yield_curve, core_cpi, pce_inflation, m2, industrial_prod,
                   retail_sales, housing_starts, consumer_sent) or a raw FRED series
                   ID (e.g. "GDP", "CPIAUCSL", "DGS10").
        observation_start: Optional ISO date "YYYY-MM-DD" — how far back to fetch.
                            Defaults to no lower bound (FRED returns latest `limit`).
        observation_end: Optional ISO date "YYYY-MM-DD" — most recent date.
        limit: Max number of observations to return (default 60, capped by FRED at 100000).

    Returns:
        Formatted string with the series data and a trend summary.
    """
    return _fred_fetch(series_id, observation_start, observation_end, limit)
