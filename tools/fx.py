"""
tools/fx.py — currency exchange rates via yfinance FX pairs.

Deliberately a RATE tool, not a converter. The agent fetches the rate here and
then multiplies with calculator(metric="multiply", ...), so every piece of
arithmetic still goes through the one place that does arithmetic (and the
calculation-fidelity check still applies to it). It is also only meant to be
used when the user explicitly asks to convert: yfinance returns every company in
its own currency and nothing in this project converts implicitly.
"""
from __future__ import annotations

import re
from typing import Optional

import yfinance as yf

_CODE = re.compile(r"^[A-Z]{3}$")


def _latest_close(symbol: str) -> Optional[tuple[float, str]]:
    """(last close, its date) for a Yahoo FX symbol such as USDINR=X, or None."""
    hist = yf.Ticker(symbol).history(period="5d")
    if hist is None or hist.empty or "Close" not in hist:
        return None
    closes = hist["Close"].dropna()
    if closes.empty:
        return None
    value = float(closes.iloc[-1])
    if value <= 0:
        return None
    return value, str(closes.index[-1])[:10]


def fx_rate(from_currency: str, to_currency: str) -> str:
    """How many units of `to_currency` one unit of `from_currency` buys."""
    frm = (from_currency or "").strip().upper()
    to = (to_currency or "").strip().upper()
    if not _CODE.match(frm) or not _CODE.match(to):
        return ("[FX] Error: currencies must be 3-letter ISO codes such as "
                "USD, INR, EUR, GBP.")
    if frm == to:
        return f"[FX] 1 {frm} = 1.000000 {to}  (same currency, no conversion needed)"

    try:
        direct = _latest_close(f"{frm}{to}=X")
        if direct:
            rate, day = direct
            return (f"[FX] 1 {frm} = {rate:.6f} {to}  "
                    f"(yfinance {frm}{to}=X, close of {day}; a market reference "
                    f"rate, not a transaction rate)")
        # Not every pair is quoted both ways — fall back to the inverse.
        inverse = _latest_close(f"{to}{frm}=X")
        if inverse:
            rate, day = 1.0 / inverse[0], inverse[1]
            return (f"[FX] 1 {frm} = {rate:.6f} {to}  "
                    f"(inverted from yfinance {to}{frm}=X, close of {day}; a market "
                    f"reference rate, not a transaction rate)")
    except Exception as exc:
        return f"[FX] Error: could not fetch the {frm}/{to} rate ({exc})."

    return f"[FX] Error: no exchange-rate data found for {frm}/{to}."
