"""
tools/chart_tool.py  –  Lets the chat agent request an inline chart.

Unlike every other tool in tools/, this doesn't fetch anything external — it's
how the agent tells the CHAT LAYER "render this data as a chart" instead of
only a markdown table. A margin comparison across 3 companies, or a revenue
trend across 5 years, is genuinely clearer as a bar/line chart than as text.

Two-part design, deliberately:
  1. The tool's return value TO THE LLM is a tiny confirmation string. The
     model doesn't need — and shouldn't burn context on — the chart's own
     data echoed back to it.
  2. The FULL structured spec (labels, series, values) is captured by
     chat_tool_node from the same tool call's `args` (already recorded in
     full, unlike `content`, which gets trimmed — see nodes/chat_agent.py's
     _run_one) and streamed to the frontend as a dedicated `chart_data` SSE
     event by api/server.py, which Chart.js renders client-side.

This keeps chart data OUT of the conversation's token budget entirely, which
matters more here than for a typical tool: a chart with 20 data points would
otherwise re-enter every subsequent LLM call in this turn and every future
turn until it aged out of the history window.
"""

from __future__ import annotations

from langchain_core.tools import tool

# Referenced by nodes/chat_agent.py (to special-case trace emission) and
# api/server.py (to decide when to emit the chart_data SSE event) — one
# constant, so the tool's registered name can never drift out of sync
# between the three places that need to agree on it.
CHART_TOOL_NAME = "render_chart"


@tool
def render_chart(
    chart_type: str,
    title: str,
    labels: list[str],
    series: list[dict],
) -> str:
    """
    Renders an inline chart in the chat UI. Use this whenever a comparison or
    trend is easier to read as a chart than a table.

    chart_type: "bar" or "line". Use "bar" for comparing entities (companies,
        metrics side by side); use "line" for a trend over time.
    title: short chart title, e.g. "Gross Margin by Company".
    labels: the x-axis categories, e.g. ["NVDA", "AMD", "MSFT"] for a
        comparison, or ["FY2022", "FY2023", "FY2024"] for a trend.
    series: one or more {"name": str, "values": [numbers, same length and
        order as labels]}. Use one series per metric, e.g.
        [{"name": "Gross Margin %", "values": [75.0, 47.0, 68.0]}].
        Multiple series render as grouped bars / multiple lines.

    Only call this with REAL numbers you already have from another tool or
    a calculator result earlier THIS turn — never invent chart data, and
    never call this before you've actually fetched the numbers it needs.

    Example:
        render_chart(
            chart_type="bar", title="Gross Margin by Company",
            labels=["NVDA", "AMD", "MSFT"],
            series=[{"name": "Gross Margin %", "values": [75.0, 47.0, 68.0]}],
        )
    """
    n_labels = len(labels or [])
    n_series = len(series or [])
    n_points = sum(len(s.get("values", []) or []) for s in (series or []))

    if n_labels == 0 or n_series == 0:
        return "[Chart] Error: labels and series cannot be empty."

    mismatched = [
        s.get("name", "?") for s in series if len(s.get("values", []) or []) != n_labels
    ]
    if mismatched:
        return (
            f"[Chart] Error: series {mismatched} have a different length than "
            f"labels ({n_labels} labels). Every series' values must match "
            f"labels one-to-one."
        )

    return f"[Chart] Rendered '{title}' ({chart_type}, {n_series} series × {n_labels} points, {n_points} values total)."
