"""
tools/earnings_transcript.py  –  Earnings call transcript via Alpha Vantage.

Correct API format per official docs:
  https://www.alphavantage.co/documentation/#transcript

  Required params:
    function  = EARNINGS_CALL_TRANSCRIPT
    symbol    = IBM
    quarter   = 2024Q1          ← format is YYYYQm  (e.g. 2024Q1, 2024Q4)
    apikey    = YOUR_KEY

  The quarter parameter is always  {YEAR}Q{1|2|3|4}
  e.g.  year=2024, quarter="Q3"  →  "2024Q3"
        year=2024, quarter=3     →  "2024Q3"
        year=2024, quarter="3"   →  "2024Q3"
"""

from __future__ import annotations
import os
import requests
from dotenv import load_dotenv

load_dotenv()


def _build_quarter_param(quarter: str | int, year: int) -> str:
    """
    Converts any quarter input to the Alpha Vantage required format: YYYYQm

    Handles all cases the LLM might send:
        ("Q1", 2024)  → "2024Q1"
        ("q4", 2024)  → "2024Q4"
        (4,    2024)  → "2024Q4"
        ("4",  2024)  → "2024Q4"
        ("2024Q3", *)  → "2024Q3"  (already formatted, pass through)
    """
    q = str(quarter).strip().upper()

    # Already fully formatted like "2024Q1" — pass through
    if len(q) == 6 and q[4] == "Q" and q[:4].isdigit():
        return q

    # Strip leading "Q" if present: "Q3" → "3"
    if q.startswith("Q"):
        q = q[1:]

    # Now q should be just the digit: "1", "2", "3", "4"
    return f"{year}Q{q}"


def earnings_transcript(ticker: str, quarter: str | int, year: int) -> str:
    """
    Retrieves the earnings call transcript for a specific company,
    quarter, and year via Alpha Vantage.

    Args:
        ticker:   Stock ticker, e.g. "AAPL", "MSFT".
        quarter:  Quarter as string OR int.
                  Accepts: "Q1","Q2","Q3","Q4" or 1,2,3,4 or "1","2","3","4"
        year:     Calendar year as int, e.g. 2024.

    Returns:
        Full transcript text with speaker labels and sentiment tags,
        or a descriptive error/skip message the agent can handle.
    """
    av_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not av_key:
        return "[Transcript] Error: ALPHA_VANTAGE_KEY environment variable is missing."

    ticker           = ticker.upper().strip()
    # ── Correct format: YYYYQm per Alpha Vantage docs ────────────────────────
    formatted_quarter = _build_quarter_param(quarter, year)

    url    = "https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol":   ticker,
        "quarter":  formatted_quarter,    # e.g. "2024Q4"
        "apikey":   av_key,
    }

    print(f"[Transcript] Fetching {ticker} {formatted_quarter} from Alpha Vantage...")

    try:
        # Share the same rate-limit pacing as financial_data.py
        from tools.financial_data import _AV_MIN_INTERVAL, _AV_LAST_CALL_TIME
        import time as _time
        import tools.financial_data as _fd
        elapsed = _time.time() - _fd._AV_LAST_CALL_TIME
        if elapsed < _AV_MIN_INTERVAL:
            wait = _AV_MIN_INTERVAL - elapsed
            print(f"  [AV rate-limit] waiting {wait:.1f}s before transcript call...")
            _time.sleep(wait)

        response = requests.get(url, params=params, timeout=20)
        _fd._AV_LAST_CALL_TIME = _time.time()
        response.raise_for_status()
        data = response.json()

        # ── API-level guards ──────────────────────────────────────────────────
        if "Information" in data:
            return (
                f"[Transcript] Alpha Vantage rate limit or plan restriction: "
                f"{data['Information']}"
            )
        if "Error Message" in data:
            return (
                f"[Transcript] No transcript found for "
                f"{ticker} {formatted_quarter}. "
                f"Error: {data['Error Message']}"
            )
        if not data:
            return f"[Transcript] Empty response for {ticker} {formatted_quarter}."

        transcript_list = data.get("transcript")
        if not transcript_list:
            # Log raw keys to help diagnose unexpected response shapes
            keys = list(data.keys())
            return (
                f"[Transcript] No 'transcript' key in response for "
                f"{ticker} {formatted_quarter}. "
                f"Response keys: {keys}"
            )

        # ── Format transcript ─────────────────────────────────────────────────
        lines = [
            f"[Transcript] {ticker} | {formatted_quarter}",
            "=" * 60,
        ]

        for turn in transcript_list:
            speaker = turn.get("speaker", "Unknown Speaker")

            # Alpha Vantage uses different keys on free vs paid tiers
            speech = (
                turn.get("speech")
                or turn.get("text")
                or turn.get("content")
                or turn.get("speech_text")
                or ""
            ).strip()

            # Sentiment block
            sentiment_obj = turn.get("sentiment_analysis", {})
            if isinstance(sentiment_obj, dict):
                sentiment = sentiment_obj.get("sentiment", "Neutral")
            else:
                sentiment = str(sentiment_obj) if sentiment_obj else "Neutral"

            if not speech:
                speech = "[Speech content not available on current API tier]"

            lines.append(f"\n[{speaker}] (Sentiment: {sentiment}):\n{speech}")

        full_text = "\n".join(lines)

        # Cap at 10 000 chars to avoid context overflow
        if len(full_text) > 10_000:
            full_text = full_text[:10_000] + "\n\n[Transcript truncated at 10 000 chars]"

        return full_text

    except requests.exceptions.Timeout:
        return f"[Transcript] Timeout fetching {ticker} {formatted_quarter}."
    except requests.exceptions.HTTPError as exc:
        return f"[Transcript] HTTP error {exc.response.status_code} for {ticker} {formatted_quarter}."
    except Exception as exc:
        return f"[Transcript] Unexpected error: {exc}"