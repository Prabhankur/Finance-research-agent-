"""
utils/result_formatter.py  –  Renders retrieved tool data WITHOUT an LLM.

The point of this module: when the final synthesis LLM call fails, the data
that real API calls already fetched (and already paid for, against a daily
quota) must not be thrown away. This turns accumulated structured tool
results into something a human can read, using pure Python — no model call,
so it cannot fail for the same reason the synthesis just did.

Deliberately NOT dumping raw JSON at the user. Raw provider payloads are
enormous, full of nulls, and use field names like
`RevenueFromContractWithCustomerExcludingAssessedTax`. The formatter picks
out the lines that matter and scales the numbers, so
`{"revenue": 331839000000}` reads as `Revenue: $331.84B`.
"""

from __future__ import annotations

from typing import Any


# Turn-level outcome states. Four states rather than a binary
# success/error, because "we fetched everything and then failed to write it
# up" and "we couldn't fetch anything" are genuinely different situations
# that deserve different messages — and, later, different retry behaviour
# (synthesis can be retried for free; retrieval cannot).
COMPLETE = "COMPLETE"
RETRIEVAL_PARTIAL = "RETRIEVAL_PARTIAL"
RETRIEVAL_COMPLETE_SYNTHESIS_FAILED = "RETRIEVAL_COMPLETE_SYNTHESIS_FAILED"
RETRIEVAL_FAILED = "RETRIEVAL_FAILED"


def format_number(value: Any) -> str:
    """Scales a raw financial figure to a human unit ($331.84B, 45.10%)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    magnitude = abs(number)
    if magnitude >= 1e12:
        return f"${number / 1e12:.2f}T"
    if magnitude >= 1e9:
        return f"${number / 1e9:.2f}B"
    if magnitude >= 1e6:
        return f"${number / 1e6:.2f}M"
    if magnitude >= 1e3:
        return f"${number:,.0f}"
    return f"{number:,.2f}"


def _summarise_text_result(text: str, max_lines: int = 14) -> list[str]:
    """
    Extracts the most informative lines from a tool's text output.

    Tools here return formatted text blocks, not JSON, so this prefers lines
    that actually carry a value (contain a colon and a digit) over headers
    and separator rules — that's what makes the difference between showing
    the user 14 lines of `====` and 14 lines of real figures.
    """
    lines = [line.rstrip() for line in str(text).splitlines()]
    valued = [
        line for line in lines
        if ":" in line
        and any(char.isdigit() for char in line)
        and not line.strip().startswith(("=", "-", "#"))
    ]
    chosen = valued[:max_lines] if valued else [
        line for line in lines if line.strip()
    ][:max_lines]
    return [line.strip() for line in chosen]


def format_tool_result(tool: str, args: dict, result: Any, max_lines: int = 14) -> str:
    """Renders one successful tool call as a readable markdown block."""
    label_bits = []
    for key in ("ticker", "query", "series_id", "series", "metric", "cik"):
        if args.get(key):
            label_bits.append(f"{key}={args[key]}")
    label = f" ({', '.join(label_bits)})" if label_bits else ""

    header = f"**{tool}**{label}"

    if isinstance(result, dict):
        body_lines = [
            f"- {key}: {format_number(value) if isinstance(value, (int, float)) else value}"
            for key, value in list(result.items())[:max_lines]
        ]
    else:
        body_lines = [f"- {line}" for line in _summarise_text_result(result, max_lines)]

    if not body_lines:
        body_lines = ["- (no readable content)"]

    return header + "\n" + "\n".join(body_lines)


def build_partial_response(
    successful_results: list[dict],
    error: Exception | str,
    failed_results: list[dict] | None = None,
) -> str:
    """
    Builds the user-facing message for a turn where retrieval worked but
    synthesis did not.

    Explicitly framed as INCOMPLETE. It would be worse than useless — it
    would be misleading — to present raw retrieved figures as though they
    were a finished analysis, because the whole value the synthesis step
    adds is cross-source reconciliation and interpretation, which is
    precisely what just failed. So the message says plainly what was
    retrieved, what was not done, and why.
    """
    failed_results = failed_results or []

    detail = str(error).strip().replace("\n", " ")
    if len(detail) > 240:
        detail = detail[:240] + "…"

    parts: list[str] = [
        "### Partial result — data retrieved, analysis incomplete",
        "",
        "I fetched the data below successfully, but the final step that "
        "interprets and cross-checks it failed. **These are raw retrieved "
        "figures, not a finished analysis** — they haven't been reconciled "
        "across sources or sanity-checked.",
        "",
        f"**Retrieved successfully ({len(successful_results)} "
        f"{'call' if len(successful_results) == 1 else 'calls'}):**",
        "",
    ]

    for record in successful_results:
        parts.append(
            format_tool_result(record["tool"], record.get("args", {}), record.get("result", ""))
        )
        parts.append("")

    if failed_results:
        parts.append("**Failed during retrieval:**")
        for record in failed_results:
            parts.append(f"- {record['tool']}: {record.get('error_class', 'failed')}")
        parts.append("")

    parts += [
        "**Not completed:** cross-source interpretation, derived metrics not "
        "yet calculated, and the written analysis.",
        "",
        f"**Technical detail:** `{detail}`",
        "",
        "Asking again will reuse nothing — re-running costs another set of "
        "API calls — so if the figures above answer your question, work from "
        "them directly.",
    ]

    return "\n".join(parts)


def classify_turn(
    successful_results: list[dict],
    failed_results: list[dict],
    synthesis_failed: bool,
) -> str:
    """Maps what happened this turn onto one of the four outcome states."""
    if synthesis_failed:
        if not successful_results:
            return RETRIEVAL_FAILED
        return RETRIEVAL_COMPLETE_SYNTHESIS_FAILED
    if failed_results and successful_results:
        return RETRIEVAL_PARTIAL
    if failed_results and not successful_results:
        return RETRIEVAL_FAILED
    return COMPLETE
