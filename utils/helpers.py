"""utils/helpers.py – Shared utility functions."""
from __future__ import annotations
import re, json
from typing import Any

def extract_ticker(text: str):
    m = re.search(r"(?:ticker|symbol)[:\s]+([A-Z]{1,5})\b", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r"\(([A-Z]{1,5})\)", text)
    if m: return m.group(1).upper()
    return None

def safe_json_parse(text: str, fallback: Any = None) -> Any:
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()
    try: return json.loads(clean)
    except: return fallback

def truncate(text: str, max_chars: int = 6000) -> str:
    return text[:max_chars] + "... [truncated]" if len(text) > max_chars else text

def format_number(value: Any, prefix: str = "$") -> str:
    try:
        v = float(str(value).replace(",", ""))
        if abs(v) >= 1_000_000_000: return f"{prefix}{v/1_000_000_000:.2f}B"
        if abs(v) >= 1_000_000: return f"{prefix}{v/1_000_000:.2f}M"
        return f"{prefix}{v:,.0f}"
    except: return str(value)

def strip_html(html: str) -> str:
    import re
    return re.sub(r"\s{3,}", "\n", re.sub(r"<[^>]+>", " ", html)).strip()