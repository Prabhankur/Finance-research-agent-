"""
nodes/intent_router.py  –  First node of every chat turn.

Decides what KIND of turn this is before any tool or expensive LLM call runs:

    CONFIRM_REPORT  user is saying "yes" to a report we offered last turn
    REPORT          user wants the full multi-page document pipeline
    SMALLTALK       greeting / "what can you do" / thanks — no tools needed
    CHAT            everything else — answer conversationally, tools allowed

Design note — rules before LLM:
A classifier LLM call costs latency and a slice of the Groq TPM budget on
EVERY turn, including "hi". So obvious cases are caught by cheap regex first
and only genuinely ambiguous turns reach the model. `intent_source` records
which path fired, so when routing misbehaves you can tell instantly whether
to fix a regex or a prompt instead of guessing.

Design note — report handoff:
The report pipeline is a ~3-6 minute, ~15-API-call run. Launching it because
someone typed "tell me about Apple" is a terrible default. So REPORT does not
run anything: it stages the request in `pending_report_*` and replies with a
confirmation offer. Only an explicit yes on the NEXT turn sets
report_confirmed, which is the signal the API layer watches to actually start
the pipeline. That's the human-in-the-loop checkpoint from the handoff doc,
implemented at the routing layer rather than as a LangGraph interrupt —
same guarantee, and it survives a stateless HTTP boundary, which an
interrupt() mid-graph does not without extra plumbing.
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import (
    INTENT_CHAT,
    INTENT_CONFIRM_REPORT,
    INTENT_REPORT,
    INTENT_SMALLTALK,
    REPORT_TYPE_LABELS,
    COMPANY_RESEARCH,
)
from graph.chat_state import ChatState
from utils.groq_client import get_llm, invoke_with_retry

# ─────────────────────────────────────────────────────────────────────────────
#  Cheap rule layer
# ─────────────────────────────────────────────────────────────────────────────

# Affirmatives that count as confirming a pending report offer. Kept tight on
# purpose: "yes but only the valuation part" is NOT a clean yes, and should
# fall through to the LLM rather than launching a 6-minute run on a guess.
_AFFIRMATIVE = re.compile(
    r"^\s*(y|ya|yes+|yeah|yep|yup|sure|ok|okay|do it|go ahead|please do|"
    r"generate it|make it|build it|run it|confirm|confirmed|proceed|"
    r"sounds good|go for it)\s*[.!]*\s*$",
    re.IGNORECASE,
)

_NEGATIVE = re.compile(
    r"^\s*(n|no+|nope|nah|not now|cancel|never ?mind|don'?t|stop)\s*[.!]*\s*$",
    re.IGNORECASE,
)

# Explicit document requests. These are unambiguous enough to skip the LLM.
_REPORT_PATTERNS = [
    r"\b(generate|create|make|build|write|prepare|produce|give me|i want|i need)\b[^.?!]{0,40}\b(report|deck|write[- ]?up|document|memo|research note|full analysis|detailed analysis)\b",
    r"\b(full|detailed|complete|comprehensive|in[- ]depth|deep[- ]dive)\b[^.?!]{0,25}\b(report|analysis|research)\b",
    r"\b(equity research|investment memo|research report|analyst report)\b",
    r"\bdownload(able)?\b[^.?!]{0,20}\b(report|docx|word|pdf)\b",
]
_REPORT_RE = [re.compile(p, re.IGNORECASE) for p in _REPORT_PATTERNS]

# Greetings / meta. Anchored to the whole message so "hi, what is AAPL's P/E"
# does not get classified as smalltalk and lose its tool access.
_SMALLTALK = re.compile(
    r"^\s*(hi+|hey+|hello|yo|good (morning|evening|afternoon)|thanks?|thank you|"
    r"thx|ty|bye|goodbye|cool|nice|great|got it|who are you|what (can|do) you do|"
    r"what are you|help|/help)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# Words that LOOK like tickers but aren't. Without this, "IS AAPL A BUY" yields
# a ticker of "IS", and every downstream lookup silently queries the wrong
# symbol — one of the nastiest classes of bug because nothing errors.
_TICKER_STOPWORDS = {
    "A", "I", "AI", "AN", "AND", "ARE", "AS", "AT", "BE", "BUY", "BY", "CAGR",
    "CAN", "CEO", "CFO", "DCF", "DO", "EPS", "EV", "FOR", "FY", "GDP", "GET",
    "HOW", "I", "IF", "IN", "IS", "IT", "ITS", "ME", "MY", "NO", "NOT", "OF",
    "OK", "ON", "OR", "PE", "Q1", "Q2", "Q3", "Q4", "ROE", "ROA", "SEC", "SO",
    "THE", "TO", "US", "USA", "VS", "WHAT", "WHEN", "WHY", "WITH", "YES", "YOY",
}

# Common company name -> ticker, so "how is apple doing" carries context even
# though the user never typed a symbol. Small on purpose: this is a
# convenience layer, not a security master. Unknown names simply fall through
# and the agent resolves them with a tool call.
_NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "tesla": "TSLA", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "walmart": "WMT", "disney": "DIS", "boeing": "BA", "coca cola": "KO",
    "pepsi": "PEP", "nike": "NKE", "starbucks": "SBUX", "salesforce": "CRM",
    "oracle": "ORCL", "adobe": "ADBE", "uber": "UBER", "airbnb": "ABNB",
    "palantir": "PLTR", "jpmorgan": "JPM", "goldman": "GS", "visa": "V",
    "mastercard": "MA", "infosys": "INFY", "tcs": "TCS.NS", "reliance": "RELIANCE.NS",
    "hdfc": "HDFCBANK.NS", "wipro": "WIT", "broadcom": "AVGO", "qualcomm": "QCOM",
}


def extract_ticker_from_text(text: str) -> Optional[str]:
    """
    Best-effort ticker extraction for sticky conversational context.

    Returns None rather than guessing. A wrong ticker is strictly worse than
    no ticker: the agent will cheerfully fetch real, correct data for the wrong
    company and present it with full confidence, and nothing downstream flags
    it. No ticker just means the agent asks, or resolves it with a tool call.

    Two rules that exist because of specific false positives caught in tests:

    1. EARLIEST match wins, across both symbols and company names. Returning
       the first dict hit meant "compare TCS and Infosys" resolved to whichever
       name sat earlier in the dict literal — an ordering bug that produces a
       confidently wrong answer with no error anywhere.

    2. Bare symbols must be 2-5 chars. A 1-char rule matched the "P" in "P/E",
       which then stuck as the conversation's active ticker for every later
       turn. Cost: genuine single-letter tickers (V, F) aren't auto-detected
       from a bare mention — fair trade, and "$V" still works below.
    """
    candidates: list[tuple[int, str]] = []

    # Explicit $SYMBOL is unambiguous — highest confidence, return immediately.
    explicit = re.search(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?)\b", text)
    if explicit:
        return explicit.group(1).upper()

    lowered = text.lower()
    for name, ticker in _NAME_TO_TICKER.items():
        match = re.search(rf"\b{re.escape(name)}\b", lowered)
        if match:
            candidates.append((match.start(), ticker))

    # Bare uppercase symbols. Lookarounds stop matches inside abbreviations
    # like "P/E", "S&P" or "U.S.".
    for match in re.finditer(
        r"(?<![A-Za-z0-9/&.$])([A-Z]{2,5}(?:\.[A-Z]{1,2})?)(?![A-Za-z0-9/])", text
    ):
        token = match.group(1).upper()
        if token not in _TICKER_STOPWORDS:
            candidates.append((match.start(), token))
            break

    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[0])[1]


# Aliases a user might type to OVERRIDE the classifier's guess while a report
# offer is pending — e.g. the classifier said COMPANY_RESEARCH but the user
# actually wants VALUATION_RISK. Matched as a substring search, not anchored,
# so both a bare "valuation risk please" and a full sentence like "generate
# it as a Valuation / Risk Deep-Dive report" resolve the same way. This is
# also what the web UI's report-type buttons send under the hood (see
# api/static/index.html) — button and free-typed text share one code path.
_TYPE_ALIASES: dict[str, list[str]] = {
    "COMPANY_RESEARCH":     ["company research", "company / stock research", "stock research", "company profile"],
    "COMPARATIVE_ANALYSIS": ["comparative analysis", "comparison report", "side-by-side", "side by side"],
    "EARNINGS_EVENT":       ["earnings event", "earnings / event", "earnings analysis", "event analysis"],
    "VALUATION_RISK":       ["valuation", "risk deep-dive", "valuation/risk", "valuation risk", "valuation / risk"],
    "MACRO_COUNTRY":        ["macro", "market/macro", "market / macro", "country"],
    "OTHER":                ["custom analysis", "other / custom"],
}


def _match_report_type_override(text: str) -> Optional[str]:
    """Returns a REPORT_TYPES constant if the text names one, else None."""
    lowered = text.lower()
    for report_type, aliases in _TYPE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return report_type
    return None




def _rule_intent(
    text: str, has_pending_report: bool, type_override: Optional[str] = None
) -> Optional[str]:
    """Returns an intent if the rules are confident, else None -> ask the LLM."""
    stripped = text.strip()

    if has_pending_report:
        # Naming a type — by button click or free text — IS confirmation.
        # Checked before the affirmative/negative patterns because "generate
        # it as a Valuation / Risk Deep-Dive report" doesn't match a bare
        # "yes" but is unambiguously a confirm-with-override.
        if type_override:
            return INTENT_CONFIRM_REPORT
        if _AFFIRMATIVE.match(stripped):
            return INTENT_CONFIRM_REPORT
        if _NEGATIVE.match(stripped):
            return INTENT_CHAT      # declined — just carry on chatting

    if _SMALLTALK.match(stripped):
        return INTENT_SMALLTALK

    for pattern in _REPORT_RE:
        if pattern.search(stripped):
            return INTENT_REPORT

    # A message that names a company and carries no report keyword is a chat
    # question. This is by far the most common turn type, so routing it by
    # rule takes a whole LLM call off the hot path — the difference between
    # "AAPL revenue?" answering in ~2s and in ~4s.
    if extract_ticker_from_text(text):
        return INTENT_CHAT

    # Short factual questions are overwhelmingly CHAT. Anything longer or
    # oddly phrased goes to the LLM for a real decision.
    if len(stripped) < 120 and stripped.endswith("?"):
        return INTENT_CHAT

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  LLM fallback layer
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_PROMPT = f"""You classify one message in an ongoing financial-assistant
conversation. Output ONLY one word, nothing else.

{INTENT_CHAT}: the user wants an answer in the conversation. Questions about
  numbers, metrics, filings, news, comparisons, opinions, explanations — even
  detailed ones — are CHAT as long as the user wants a REPLY, not a document.

{INTENT_REPORT}: the user is explicitly asking for a generated document /
  report / write-up they can download or read as a formal deliverable.

{INTENT_SMALLTALK}: greeting, thanks, goodbye, or a question about what the
  assistant itself can do. No financial data needed.

EXAMPLES
"what's Apple's P/E right now" -> {INTENT_CHAT}
"compare NVDA and AMD margins" -> {INTENT_CHAT}
"why did Tesla stock drop last week" -> {INTENT_CHAT}
"walk me through Apple's cash flow" -> {INTENT_CHAT}
"is NVDA overvalued, explain properly" -> {INTENT_CHAT}
"generate a full research report on Apple" -> {INTENT_REPORT}
"I need a detailed write-up on TSLA I can share" -> {INTENT_REPORT}
"make me an equity research note for MSFT" -> {INTENT_REPORT}
"hey" -> {INTENT_SMALLTALK}
"what can you do" -> {INTENT_SMALLTALK}

RULE: when unsure between CHAT and REPORT, choose {INTENT_CHAT}. Answering in
chat is cheap and reversible; launching a report run is neither."""


def _llm_intent(text: str) -> str:
    llm = get_llm(temperature=0.0, max_tokens=12)
    try:
        response = invoke_with_retry(
            llm,
            [SystemMessage(content=_INTENT_PROMPT), HumanMessage(content=text)],
        )
        raw = str(response.content).strip().upper()
    except Exception as exc:
        print(f"[IntentRouter] LLM classification failed ({exc}) -> CHAT")
        return INTENT_CHAT

    for intent in (INTENT_REPORT, INTENT_SMALLTALK, INTENT_CHAT):
        if intent in raw:
            return intent
    return INTENT_CHAT


def _classify_report_type(query: str, ticker: Optional[str]) -> str:
    """
    Reuses the report pipeline's own classifier so the type shown in the chat
    confirmation ("I'll build a COMPARATIVE_ANALYSIS report") is the same type
    the pipeline will actually build. Importing the node instead of copying
    its prompt means there is exactly one definition of this logic.

    BUG FIXED: this used to call the classifier with a state dict missing the
    "ticker" key entirely. report_type_classifier_node's own exception fallback
    is `COMPANY_RESEARCH if state.get("ticker") else OTHER` — with no ticker
    key present, `state.get("ticker")` is always falsy, so ANY failure of the
    classifier's LLM call (e.g. a Groq rate limit, which is exactly what was
    happening in production) silently produced OTHER even when the query
    plainly named a company ("analyze Microsoft"). The chat agent's own
    ticker extraction already runs earlier in this node — passing it through
    here means the fallback path is actually reachable and correct instead of
    always landing on the wrong bucket.
    """
    try:
        from nodes.report_type_classifier import report_type_classifier_node

        result = report_type_classifier_node(
            {"raw_query": query, "report_type": "", "ticker": ticker or ""}
        )
        return result.get("report_type", COMPANY_RESEARCH)
    except Exception as exc:
        print(f"[IntentRouter] Report-type classification failed ({exc})")
        from config import OTHER as _OTHER

        return COMPANY_RESEARCH if ticker else _OTHER


# ─────────────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────────────

def intent_router_node(state: ChatState) -> dict:
    messages = state.get("messages", [])

    # The current turn is the last human message on the thread.
    user_input = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            user_input = str(message.content)
            break

    pending_query = state.get("pending_report_query")
    has_pending = bool(pending_query)

    # Ticker is computed BEFORE classification so it can feed both the sticky
    # conversation context AND the report-type classifier's fallback (see the
    # bug note on _classify_report_type above).
    ticker = extract_ticker_from_text(user_input) or state.get("active_ticker")

    type_override = _match_report_type_override(user_input) if has_pending else None

    intent = _rule_intent(user_input, has_pending, type_override)
    source = "rules"
    if intent is None:
        intent = _llm_intent(user_input)
        source = "llm"

    # Per-turn fields are reset here, at the single entry point of the turn.
    # Doing it here rather than in each node means there is one place to look
    # when a value unexpectedly survives from a previous turn.
    update: dict = {
        "user_input": user_input,
        "intent": intent,
        "intent_source": source,
        "tool_loops": 0,
        "active_ticker": ticker,
        "report_confirmed": False,
        # Both use the reset_or_append reducer (see graph/chat_state.py):
        # passing None clears them, so each turn starts with a clean slate.
        # Without this they'd grow for the life of the thread, and a
        # synthesis failure would replay tool results fetched for unrelated
        # earlier questions as though they answered the current one.
        "successful_tool_results": None,
        "tool_trace": None,
        "contradictions_detected": None,
        "turn_status": "",
    }

    print(
        f"[IntentRouter] intent={intent} via={source} ticker={ticker} "
        f"type_override={type_override}"
    )

    if intent == INTENT_CONFIRM_REPORT:
        # Keep pending_report_query — the API reads it to launch the pipeline.
        # A type_override replaces the classifier's original guess: this is
        # the "user picked a different report type" path, whether they typed
        # it or clicked a button in the web UI (see api/static/index.html).
        update["report_confirmed"] = True
        if type_override:
            update["pending_report_type"] = type_override
            print(f"[IntentRouter] Report type overridden to {type_override}")
        return update

    if intent == INTENT_REPORT:
        report_type = _classify_report_type(user_input, ticker)
        label = REPORT_TYPE_LABELS.get(report_type, report_type)

        # Every other supported type, so the user can override in one line
        # instead of guessing what's available. The web UI renders these as
        # clickable buttons instead (same override mechanism underneath —
        # see _match_report_type_override).
        other_lines = "\n".join(
            f"  • {other_label}"
            for other_type, other_label in REPORT_TYPE_LABELS.items()
            if other_type != report_type
        )

        update["pending_report_query"] = user_input
        update["pending_report_type"] = report_type
        update["messages"] = [
            AIMessage(
                content=(
                    f"That's a full-report request. I'd run this as a "
                    f"**{label}** report — it fetches filings, financials and "
                    f"news, computes the metrics, and produces a Markdown + "
                    f"Word document with charts.\n\n"
                    f"It takes a few minutes and uses a chunk of the daily "
                    f"API budget, so confirm before I start: reply **yes** to "
                    f"go with {label}, name a different type below, or just "
                    f"ask me the question directly and I'll answer here in "
                    f"chat instead.\n\n"
                    f"Other report types available:\n{other_lines}"
                )
            )
        ]
        return update

    # CHAT / SMALLTALK — clear any stale pending offer so a later bare "yes"
    # can't resurrect a report request from three turns ago.
    update["pending_report_query"] = None
    update["pending_report_type"] = None
    return update


# ─────────────────────────────────────────────────────────────────────────────
#  Conditional edge
# ─────────────────────────────────────────────────────────────────────────────

def route_after_intent(state: ChatState) -> str:
    intent = state.get("intent", INTENT_CHAT)
    if intent == INTENT_REPORT:
        return "end"            # confirmation message already emitted
    if intent == INTENT_CONFIRM_REPORT:
        return "handoff"        # emit a "starting" message, API takes over
    if intent == INTENT_SMALLTALK:
        return "smalltalk"      # one cheap LLM call, no tools bound
    return "agent"
