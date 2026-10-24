"""
utils/error_classifier.py – Classifies a tool's returned string into one of
four categories so the executors know what to do about it, instead of
blindly marking every tool call "done" the moment something came back.

Categories:
    SYSTEMIC     – the tool itself is broken for the rest of this run
                   (missing dependency, auth failure, persistent rate limit).
                   Retrying with different args won't help. → blacklist the
                   tool and escalate to the replanner.
    BAD_ARGS     – the tool is fine, the call was malformed (wrong/missing
                   argument). → retry locally with corrected args, no replan.
    LEGITIMATE_EMPTY – a real, valid "there's nothing here" answer (e.g. no
                   filings found for this ticker/year). → don't retry, don't
                   replan, just tag it so the report can say "not available"
                   honestly instead of treating absence-of-data as a failure.
    TRANSIENT    – one-off network hiccup / timeout. → one quick same-args
                   retry before anything else escalates.
    OK           – the call actually succeeded.
"""
from __future__ import annotations
import re

SYSTEMIC = "systemic"
BAD_ARGS = "bad_args"
LEGITIMATE_EMPTY = "legitimate_empty"
TRANSIENT = "transient"
OK = "ok"

# Order matters — checked top to bottom, first match wins.
#
# IMPORTANT: none of these patterns match a bare number on its own (e.g.
# \b401\b). Financial data is full of decimals, moving averages, and
# ratios that can start with any 3-digit sequence — a real production bug
# here was "401.41074" (a 200-day moving average) matching \b401\b and
# getting misclassified as an HTTP 401 Unauthorized error, which
# blacklisted a tool that had actually succeeded. Every numeric pattern
# below requires explicit textual context (e.g. "http", "status code",
# "unauthorized") immediately around the number, so a plain decimal in
# real data can't trip it.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (SYSTEMIC, re.compile(r"not installed|no module named|modulenotfounderror", re.I)),
    (SYSTEMIC, re.compile(r"unauthorized|invalid api key|authentication (failed|error)|\bforbidden\b", re.I)),
    (SYSTEMIC, re.compile(r"http\s*(error\s*)?\b40[13]\b|status\s*code[:\s]*\b40[13]\b|\b40[13]\b\s*(error)?\s*[:\-]?\s*(unauthorized|forbidden)", re.I)),
    (SYSTEMIC, re.compile(r"api[_ ]?key.{0,15}(is )?missing|missing.{0,15}api[_ ]?key|"
                           r"environment variable is missing|credential", re.I)),
    (SYSTEMIC, re.compile(r"too many requests|rate limit exceeded|rate.limited|"
                           r"http\s*(error\s*)?\b429\b|status\s*code[:\s]*\b429\b", re.I)),
    (TRANSIENT, re.compile(r"timed? ?out|timeout|connection (reset|error|refused)", re.I)),
    (TRANSIENT, re.compile(r"http\s*(error\s*)?\b5\d\d\b|status\s*code[:\s]*\b5\d\d\b|"
                            r"server error|internal server error|bad gateway|service unavailable|gateway timeout", re.I)),
    (BAD_ARGS, re.compile(r"missing or wrong argument|missing \d+ required (positional|keyword)|unexpected keyword argument|unknown metric", re.I)),
    (BAD_ARGS, re.compile(r"tool call validation failed|was not in request\.tools|"
                           r"did not call .* as forced", re.I)),
    (LEGITIMATE_EMPTY, re.compile(r"no .*filings? found|could not resolve cik|non-us ticker", re.I)),
    (LEGITIMATE_EMPTY, re.compile(r"^n/?a$", re.I)),  # anchored to the WHOLE string, not "ends with"
]

# Any string of the form "[ToolName] Error: ..." or "[ToolName] Error ..." OR
# "[Error] ..." (error named inside the brackets, not after them) that didn't
# match a specific pattern above still gets treated as a failure rather than
# silently passing as success — default to bad_args (cheapest to recover
# from) rather than escalating on something unrecognized. This is a generic
# prefix match (any bracketed tag), not a hardcoded tool list, so it doesn't
# silently miss a tool whose exact error format wasn't seen during testing —
# every current tool tags its errors this way, and any tool added later
# gets the same safety net for free.
_GENERIC_ERROR_PATTERN = re.compile(r"^\[\w[\w .]*\]\s*error\b|^\[error\]", re.I)


def classify_result(tool_name: str, result: str) -> str:
    """Returns one of SYSTEMIC / BAD_ARGS / LEGITIMATE_EMPTY / TRANSIENT / OK."""
    if not result or not result.strip():
        return LEGITIMATE_EMPTY

    text = result.strip()
    lower = text.lower()

    for category, pattern in _PATTERNS:
        if pattern.search(text):
            return category

    # If it starts with a bracketed "[ToolName] Error..." tag but didn't
    # match a specific pattern, still treat as an error rather than success.
    if _GENERIC_ERROR_PATTERN.search(text):
        return BAD_ARGS

    return OK


def is_failure(category: str) -> bool:
    return category in (SYSTEMIC, BAD_ARGS, TRANSIENT)
