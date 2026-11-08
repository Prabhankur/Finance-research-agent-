"""
nodes/query_understanding.py  –  Step 1 of a chat turn: UNDERSTAND and ELABORATE.

Why this exists
---------------
The chat agent used to do two jobs in one model call: work out what the user
meant, AND decide which tools to call. Splitting them makes each easier, and it
fixes problems that came from the router knowing only "one ticker":

  * a question about 8 companies was reduced to the first one (state held a
    single ticker) and the prompt then told the model the topic was that company;
  * a question about a country, a person or an unlisted company had no place to
    live in state at all;
  * a follow-up inherited the previous company even when the new question was
    about something else.

This step reads the message and writes a BRIEF beside it — scope, subjects,
metrics, period, assumptions, and (only if truly blocked) one question for the
user. The user's own words are NEVER rewritten or replaced: the agent sees the
original message exactly as typed plus the brief. A rewrite can silently change
what was asked; a brief cannot.

Division of labour
------------------
  The MODEL does the understanding (that is judgement).
  The CODE enforces what happens next (that is policy):
    - the model proposes a question to ask; code honours it only when the
      question is genuinely blocked (see `_is_blocking`);
    - the model proposes tickers; code trusts its own name table over them and
      marks the rest unverified;
    - any malformed output is discarded and the rules-only router takes over.

Failure posture
---------------
Every function here returns None / a safe default instead of raising. The router
falls back to its rules, so this step can only ever improve a turn, never break one.
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config import FRONT_DOOR_MAX_TOKENS

ENTITY_TYPES = {"company", "index", "country", "person", "commodity", "currency", "other"}
# Up to 20 characters: Indian tickers with an exchange suffix are long ("BAJFINANCE.NS" is 13).
_TICKER_OK = re.compile(r"^\^?[A-Z][A-Z0-9.\-=]{0,19}$")
_BRIEF_INTENTS = {"chat", "report", "smalltalk"}
MAX_ENTITIES = 12

# Gaps that are never worth interrupting the user for: there is always a sensible
# default ("latest"), so the right move is to assume it and say so.
_DEFAULTABLE = re.compile(r"year|period|date|time|quarter|when|range|horizon|frequency|currency|unit", re.I)

FRONT_DOOR_PROMPT = """You read ONE message sent to a financial research assistant and
write a brief for the assistant that will answer it. You do NOT answer the question
and you do NOT rewrite it.

Return ONLY a JSON object with exactly these keys:
{"in_scope": true,
 "intent": "chat",
 "entities": [{"name": "Apple", "type": "company", "ticker": "AAPL"}],
 "metrics": ["market cap", "P/E"],
 "period": "latest",
 "uses_previous_context": false,
 "missing": [],
 "ask": null,
 "assumptions": ["latest available values"]}

Rules:
- in_scope: true for companies, markets, filings, earnings, valuation, economics and
  macro data, investing concepts and personal finance. false for unrelated topics
  (sports, recipes, coding, trivia). If unsure, true.
- intent: "report" ONLY if the user explicitly asks for a downloadable report,
  write-up or memo. "smalltalk" for greetings, thanks, or what you can do.
  Otherwise "chat".
- entities: every company, index, country, person, commodity or currency the
  question is about. type is one of company|index|country|person|commodity|
  currency|other. Give "ticker" ONLY for a company or index you are sure of (US
  symbols bare like AAPL; .NS NSE India, .BO BSE India, .L London, .TO Toronto);
  otherwise null. Never guess a ticker.
- metrics: what they want, in their own words.
- period: "latest" unless they name one (e.g. "FY2024", "last 5 years").
- uses_previous_context: true if the message depends on earlier turns ("its",
  "and what about", "that company", "the same one").
- missing and ask: ONLY when the question cannot be answered at all without the
  answer AND no sensible default exists (for example "compare them" with no
  companies named and no earlier context). A missing year, period or metric is
  NEVER blocking: assume a sensible default and put it in "assumptions". When
  blocked, name the gap in "missing" and write ONE short question in "ask".
- Prefer assumptions over questions."""


# ── helpers ───────────────────────────────────────────────────────────────────

def _local_name_lookup(name: str) -> Optional[str]:
    """Ticker from the router's own name table (ground truth), or None."""
    from nodes.intent_router import _NAME_TO_TICKER
    lowered = (name or "").lower()
    if lowered in _NAME_TO_TICKER:
        return _NAME_TO_TICKER[lowered]
    for known, ticker in _NAME_TO_TICKER.items():      # "Apple Inc" contains "apple"
        if re.search(rf"\b{re.escape(known)}\b", lowered):
            return ticker
    return None


def brief_tickers(brief: Optional[dict]) -> list[str]:
    """Tickers of the company / index subjects, in order, de-duplicated."""
    out: list[str] = []
    for ent in (brief or {}).get("entities") or []:
        t = ent.get("ticker")
        if t and ent.get("type") in ("company", "index") and t not in out:
            out.append(t)
    return out


def _is_blocking(ask: Optional[str], entities: list[dict], uses_previous: bool,
                 active_tickers: list[str], user_input: str) -> bool:
    """
    Policy: the model may PROPOSE a question; this decides whether it is asked.
    Blocking means there is no subject to work on: nothing named in the message,
    nothing recoverable from the conversation, nothing the local rules can find.
    Anything less — an unnamed year, a vague metric — is answered with a stated
    assumption rather than an interruption.
    """
    if not ask:
        return False
    if entities:
        return False
    if uses_previous and active_tickers:
        return False
    from nodes.intent_router import extract_tickers_from_text
    if extract_tickers_from_text(user_input):
        return False
    return True


def validate_brief(raw, user_input: str, active_tickers: Optional[list[str]] = None) -> Optional[dict]:
    """Normalises whatever the model returned into a safe brief, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    active_tickers = list(active_tickers or [])

    in_scope = raw.get("in_scope")
    in_scope = True if in_scope is None else bool(in_scope)
    intent = str(raw.get("intent") or "chat").strip().lower()
    if intent not in _BRIEF_INTENTS:
        intent = "chat"

    entities: list[dict] = []
    for item in (raw.get("entities") or [])[:MAX_ENTITIES]:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        name = str(item["name"]).strip()[:60]
        etype = str(item.get("type") or "other").strip().lower()
        etype = etype if etype in ENTITY_TYPES else "other"
        ticker, verified = None, False
        if etype in ("company", "index"):
            local = _local_name_lookup(name)
            proposed = str(item.get("ticker") or "").strip().upper()
            if local:
                ticker, verified = local, True          # our own table beats the model's memory
            elif proposed and _TICKER_OK.match(proposed):
                ticker = proposed                       # plausible but unconfirmed
        entities.append({"name": name, "type": etype, "ticker": ticker, "verified": verified})

    metrics = [str(m).strip()[:40] for m in (raw.get("metrics") or []) if str(m).strip()][:8]
    period = str(raw.get("period") or "latest").strip()[:40] or "latest"
    uses_previous = bool(raw.get("uses_previous_context"))
    assumptions = [str(a).strip()[:80] for a in (raw.get("assumptions") or []) if str(a).strip()][:4]
    missing = [str(m).strip()[:60] for m in (raw.get("missing") or []) if str(m).strip()][:4]
    ask = str(raw.get("ask") or "").strip()[:300] or None

    # A gap with a sensible default is an assumption, never a question.
    defaultable = [m for m in missing if _DEFAULTABLE.search(m)]
    for gap in defaultable:
        assumptions.append(f"{gap} was not specified, so use the latest available")
    missing = [m for m in missing if m not in defaultable]
    if not missing:
        ask = None                                      # nothing real is missing -> nothing to ask
    elif not ask:
        ask = "To answer that I need one more detail: " + "; ".join(missing) + "?"

    if not in_scope or intent != "chat" or not _is_blocking(ask, entities, uses_previous, active_tickers, user_input):
        ask = None

    return {
        "in_scope": in_scope, "intent": intent, "entities": entities, "metrics": metrics,
        "period": period, "uses_previous_context": uses_previous,
        "missing": missing if ask else [], "ask": ask, "assumptions": assumptions[:4],
        "source": "front_door",
    }


def rules_brief(user_input: str, active_tickers: Optional[list[str]] = None) -> dict:
    """The brief the router builds with no model call — the fallback and the test default."""
    from nodes.intent_router import extract_tickers_from_text
    found = extract_tickers_from_text(user_input)
    return {
        "in_scope": True, "intent": "chat",
        "entities": [{"name": t, "type": "company", "ticker": t, "verified": False} for t in found],
        "metrics": [], "period": "latest", "uses_previous_context": False,
        "missing": [], "ask": None, "assumptions": [], "source": "rules",
    }


def brief_for_prompt(brief: Optional[dict]) -> str:
    """The short block added to the agent's system prompt. Empty when there is nothing useful to say."""
    if not brief:
        return ""
    lines: list[str] = []
    companies = [e for e in brief.get("entities") or [] if e.get("type") in ("company", "index")]
    others = [e for e in brief.get("entities") or [] if e.get("type") not in ("company", "index")]
    if companies:
        shown = ", ".join(
            f"{e['name']} ({e['ticker']}{'' if e.get('verified') else '?'})" if e.get("ticker") else f"{e['name']} (ticker unknown)"
            for e in companies
        )
        lines.append(f"Companies: {shown}")
    if others:
        lines.append("Other subjects: " + ", ".join(f"{e['name']} ({e['type']})" for e in others))
    if brief.get("metrics"):
        lines.append("Metrics: " + ", ".join(brief["metrics"]) + f" · Period: {brief.get('period') or 'latest'}")
    if brief.get("assumptions"):
        lines.append("State these assumptions in your answer: " + "; ".join(brief["assumptions"]))
    tickers = brief_tickers(brief)
    if len(tickers) >= 2:
        lines.append(f"Fetch the companies with ONE financial_data_api call: tickers={tickers}")
    if any(e.get("ticker") and not e.get("verified") for e in companies):
        lines.append("A '?' ticker is unconfirmed: check the company name in the returned data and say so if it does not match.")
    if not lines:
        return ""
    return ("\n## Question brief (the user's original wording is unchanged in the conversation)\n"
            + "\n".join(lines) + "\n")


# ── the model call ────────────────────────────────────────────────────────────

def understand_query(user_input: str, history_text: str = "", active_tickers: Optional[list[str]] = None,
                     summary: Optional[str] = None) -> Optional[dict]:
    """One small structured call. Returns a validated brief, or None on ANY failure."""
    try:
        from utils.groq_client import get_llm, invoke_with_retry
        from utils.helpers import safe_json_parse

        parts: list[str] = []
        if active_tickers:
            parts.append("Companies currently in focus: " + ", ".join(active_tickers))
        if summary:
            parts.append("Conversation summary:\n" + summary[:500])
        if history_text:
            parts.append("Earlier in this conversation (context only):\n" + history_text[:800])
        parts.append("USER MESSAGE:\n" + user_input[:1500])

        llm = get_llm(temperature=0.0, max_tokens=FRONT_DOOR_MAX_TOKENS)
        response = invoke_with_retry(
            llm,
            [SystemMessage(content=FRONT_DOOR_PROMPT), HumanMessage(content="\n\n".join(parts))],
            max_retries=2,
        )
        raw = safe_json_parse(str(response.content or ""))
        return validate_brief(raw, user_input, active_tickers)
    except Exception as exc:
        print(f"[FrontDoor] Understanding step failed ({exc}) — falling back to rules.")
        return None
