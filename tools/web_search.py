"""
tools/web_search.py  –  Web search tool using Tavily.

Directly extends the user's existing web_search() function with
better error handling and a clean string return for the agent.
"""

from __future__ import annotations
import os
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()


def web_search(
    query: str,
    num_results: int = 10,
    date_range: Optional[str] = None,
) -> str:
    """
    Performs a web search for current news, analysis, and commentary
    about a company or financial topic using Tavily.

    Args:
        query:        Search string, e.g. "AAPL Q4 2024 earnings analysis".
        num_results:  Number of results to retrieve (default 10).
        date_range:   Optional. Number of days back as a string, e.g. "7".
                      If None, Tavily searches all time.

    Returns:
        A formatted string of (URL | Title | Snippet) tuples separated
        by dashes – ready to be appended to the agent's context document.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "[WebSearch] Error: TAVILY_API_KEY is missing."

    url = "https://api.tavily.com/search"
    payload: dict = {
        "api_key":      api_key,
        "query":        query,
        "search_depth": "advanced",
        "max_results":  num_results,
        "topic":        "news",
    }
    if date_range and str(date_range).isdigit():
        payload["days"] = int(date_range)

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        data    = response.json()
        results = data.get("results", [])

        if not results:
            return f"[WebSearch] No results found for query: '{query}'"

        # Build a clean, readable string block
        lines = [f"[WebSearch] Results for: '{query}'\n{'='*60}"]
        for i, item in enumerate(results, 1):
            lines.append(
                f"\n[Result {i}]\n"
                f"  Title   : {item.get('title', 'N/A')}\n"
                f"  URL     : {item.get('url', 'N/A')}\n"
                f"  Snippet : {item.get('content', 'N/A')[:500]}"
            )
        lines.append(f"\n{'='*60}")
        return "\n".join(lines)

    except Exception as exc:
        return f"[WebSearch] Error: {exc}"
