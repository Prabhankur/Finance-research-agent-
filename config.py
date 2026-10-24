"""
config.py – Single place to tune the agent's execution/recovery behavior.

Change a number here once; every node that needs it reads from this file
instead of hardcoding its own copy.
"""

# ── Tool-call recovery budgets ────────────────────────────────────────────────
# How many times a single failed tool call is retried with the SAME tool
# (corrected args / same args again) before we give up on that call alone.
LOCAL_RETRY_LIMIT = 1

# How many times, for the ENTIRE run, we're allowed to escalate to the
# replanner node. This is a global budget shared across the fetch executor
# and the compute executor — not per-step. Once exhausted, any further
# failure is just tagged as unavailable and the run moves on.
GLOBAL_REPLAN_BUDGET = 2

# Safety cap on how many calculator/compute steps the compute executor will
# process in one run, independent of how many the plan asked for. Prevents
# a bad plan from looping forever.
COMPUTE_STEP_CAP = 10

# LangGraph recursion_limit — generous safety valve, not a tuning knob you
# should need to touch often. See main.py for the formula.
GRAPH_RECURSION_LIMIT = 100

# Tools whose inputs depend on data fetched earlier in the run (i.e. their
# arguments can't be filled in at plan time). Everything else is treated as
# a "fetch" tool and executed deterministically straight from the plan.
COMPUTE_TOOLS = {"calculator"}


# ── Report types ──────────────────────────────────────────────────────────────
# The 5 archetypes this build supports fully, plus OTHER as an escape hatch.
# Each type maps to a default section list that the roadmap_planner uses as a
# starting point — the LLM then adds/removes based on the specific query
# (e.g. "analyse X but skip risk"), so these are guidance, not constraints.
#
# Section IDs referenced here MUST also exist as keys in
# nodes/report_generator.py's SECTION_INSTRUCTIONS dict, otherwise the writer
# won't know how to render them. Adding a new section = touch both places.
#
# CUSTOM_ANALYSIS is the "free hand" slot the LLM populates when the OTHER
# type is selected — it has generic instructions that let the model address
# whatever the specific query focused on.

COMPANY_RESEARCH     = "COMPANY_RESEARCH"
COMPARATIVE_ANALYSIS = "COMPARATIVE_ANALYSIS"
EARNINGS_EVENT       = "EARNINGS_EVENT"
VALUATION_RISK       = "VALUATION_RISK"
MACRO_COUNTRY        = "MACRO_COUNTRY"
OTHER                = "OTHER"

REPORT_TYPES = [
    COMPANY_RESEARCH, COMPARATIVE_ANALYSIS, EARNINGS_EVENT,
    VALUATION_RISK, MACRO_COUNTRY, OTHER,
]

REPORT_TYPE_LABELS = {
    COMPANY_RESEARCH:     "Company / Stock Research",
    COMPARATIVE_ANALYSIS: "Comparative Analysis (2+ companies)",
    EARNINGS_EVENT:       "Earnings / Event Analysis",
    VALUATION_RISK:       "Valuation / Risk Deep-Dive",
    MACRO_COUNTRY:        "Market / Macro / Country",
    OTHER:                "Other / Custom",
}

REPORT_TYPE_DESCRIPTIONS = {
    COMPANY_RESEARCH:     "Deep analysis of a single company: financials, growth, valuation, risk, outlook. Common queries: 'analyse X', 'should I buy X', 'how is X performing'.",
    COMPARATIVE_ANALYSIS: "Side-by-side comparison of 2+ specific companies on financial, growth, valuation, and risk dimensions. Common queries: 'compare X and Y', 'X vs its competitor'.",
    EARNINGS_EVENT:       "Time-scoped analysis of a specific quarter, event, or price move. Common queries: 'analyse X's Q3 results', 'why did X drop after earnings'.",
    VALUATION_RISK:       "Narrow quantitative investigation of one company's valuation or risk profile using DCF, multiples, sensitivity, downside scenarios. Common queries: 'is X overvalued', 'how risky is X'.",
    MACRO_COUNTRY:        "Economy or market-level analysis, no primary company. Common queries: 'compare India and China growth', 'summarize today's market', 'how are semiconductors doing'.",
    OTHER:                "Query doesn't fit any of the above — the LLM will build a custom section list and include one CUSTOM_ANALYSIS section addressing the specific angle asked about.",
}

# Default section list per report type. Each list is the STARTING POINT the
# planner uses; the LLM adjusts it based on the specific query.
DEFAULT_SECTIONS_BY_TYPE = {
    COMPANY_RESEARCH: [
        "EXEC_SUMMARY", "COMPANY_PROFILE", "EARNINGS_ANALYSIS",
        "GROWTH_DRIVERS", "VALUATION", "RISK_ASSESSMENT",
        "MANAGEMENT_GUIDANCE", "INVESTMENT_THESIS", "APPENDIX",
    ],
    COMPARATIVE_ANALYSIS: [
        "EXEC_SUMMARY", "COMPANY_SNAPSHOTS", "FINANCIAL_COMPARISON",
        "GROWTH_COMPARISON", "VALUATION_COMPARISON", "RISK_COMPARISON",
        "WINNER_TRADEOFFS", "APPENDIX",
    ],
    EARNINGS_EVENT: [
        "EXEC_SUMMARY", "EVENT_SUMMARY", "MARKET_REACTION",
        "COMPANY_FACTORS", "MANAGEMENT_GUIDANCE", "NEWS_CONTEXT",
        "HISTORICAL_COMPARISON", "IMPACT_ASSESSMENT",
    ],
    VALUATION_RISK: [
        "EXEC_SUMMARY", "COMPANY_PROFILE", "VALUATION_MODELS",
        "PEER_MULTIPLES", "SENSITIVITY_ANALYSIS", "DOWNSIDE_SCENARIOS",
        "RISK_FACTORS", "CONCLUSION",
    ],
    MACRO_COUNTRY: [
        "EXEC_SUMMARY", "MACRO_INDICATORS", "MARKET_TRENDS",
        "SECTOR_PERFORMANCE", "RATES_INFLATION_GDP",
        "COUNTRY_COMPARISON", "OUTLOOK",
    ],
    OTHER: [
        "EXEC_SUMMARY", "COMPANY_PROFILE", "EARNINGS_ANALYSIS",
        "GROWTH_DRIVERS", "VALUATION", "RISK_ASSESSMENT",
        "MANAGEMENT_GUIDANCE", "INVESTMENT_THESIS",
        "MACRO_INDICATORS", "MARKET_TRENDS", "SECTOR_PERFORMANCE",
        "CUSTOM_ANALYSIS", "APPENDIX",
    ],
}


# ── Chat agent mode ──────────────────────────────────────────────────────────
# The chat agent is a SEPARATE graph from the report pipeline. Same tools,
# same Groq client, different control flow: a bounded ReAct loop that answers
# in one conversational turn instead of planning a 20-page document.
#
# Why bounded: an unbounded ReAct loop on a rate-limited free tier (AV 25/day,
# Groq TPM caps) is how you burn an API budget in one bad turn. The cap below
# is a hard stop — when it's hit the agent is re-invoked WITHOUT tools bound,
# so it's forced to answer from whatever it already gathered rather than
# looping or erroring out.

# Max agent<->tool round trips in a single chat turn.
CHAT_MAX_TOOL_LOOPS = 4

# Max tool calls the agent may issue in one round trip (fan-out width).
#
# Was 4. Raised to 8 after diagnosing a real production failure: a 3-company
# comparison ("NVDA vs MSFT vs AMD") needs financial_data_api + calculator
# PER company — 6 calls — which exceeded the old cap of 4 and forced the
# agent to split the work across two ReAct rounds instead of one. That's not
# just slower: each extra round re-sends the FULL accumulated tool output
# from the previous round back into the next LLM call, so splitting a
# 6-call comparison into two rounds roughly doubled the context consumed for
# that turn — which is exactly what was tipping some multi-entity
# comparisons over Groq's request size limit (BadRequestError).
#
# 8 covers up to ~4 entities at 2 calls each in a single round. Raising this
# further is safe from a network-concurrency standpoint (calls are I/O bound)
# but grows the same-round context linearly, so CHAT_TOOL_RESULT_MAX_CHARS
# and the token-budget check in chat_agent_node are the actual backstops —
# see _window_messages()'s token-based trimming, not this constant, for the
# real ceiling.
CHAT_MAX_PARALLEL_TOOL_CALLS = 8

# How many prior messages to carry forward per thread. Older turns are dropped
# (oldest-first) rather than summarised — summarisation is a future upgrade.
CHAT_HISTORY_MAX_MESSAGES = 24

# Hard ceiling on the TOKEN size of the message history sent to Groq, checked
# in ADDITION to CHAT_HISTORY_MAX_MESSAGES.
#
# Why both: message-count alone isn't a safe bound. A 24-message window can
# still blow past a model's request-size limit if several of those messages
# are large tool results — which is exactly what happened in production on
# multi-entity comparisons and produced 400/413 errors.
#
# Sized against the REQUEST limit, not the model's context window. A
# production 413 read: "Request too large ... on tokens per minute (TPM):
# Limit 8000, Requested 10103". On a free tier the TPM cap (8000) binds well
# before any model's context window (8k-128k) does, and the whole request —
# system prompt + history + bound tool schemas — counts against it. So the
# budget below leaves room for: ~800 tokens of system prompt, ~1200 tokens of
# JSON tool schemas for 7 bound tools, and the 2048-token completion.
CHAT_MAX_CONTEXT_TOKENS = 3500

# Per-tool-result cap. Lowered in step with CHAT_MAX_CONTEXT_TOKENS: a single
# 5000-char result is ~1250 tokens, so three of them alone would exceed the
# whole budget above. The full untrimmed payload is still preserved for the
# user via successful_tool_results (see graph/chat_state.py) — this cap only
# governs what re-enters the LLM's context.
CHAT_TOOL_RESULT_MAX_CHARS = 2500

# Tools the chat agent is allowed to call. financial/SEC/web/calculator/fred
# come from the SHARED registry (tools/__init__.py) also used by the report
# pipeline. render_chart is chat-ONLY — it's deliberately not added to
# tools/__init__.py's ALL_TOOLS, because that list also feeds the report
# pipeline's roadmap planner prompt, and "render an inline chat chart" is a
# concept the document pipeline (which renders its OWN matplotlib charts via
# utils/chart_generator.py) has no use for.
CHAT_ENABLED_TOOLS = [
    "financial_data_api",
    "web_search",
    "sec_filing_search",
    "earnings_transcript",
    "calculator",
    "fred_series",
    "render_chart",
]

# Tools that must NOT run concurrently with each other (shared rate limiter).
# earnings_transcript is Alpha Vantage, which paces at ~13s between calls.
CHAT_SERIAL_TOOLS = {"earnings_transcript"}

# ── Rolling summarization ────────────────────────────────────────────────────
# Don't bother summarizing until at least this many messages have fallen out
# of the keep-window — below this, the LLM call costs more than it saves.
# This is also what makes summarization self-limiting: right after a fold,
# the "old" portion of history is near-empty, so it naturally won't fire
# again until enough NEW messages have accumulated to cross this bar.
CHAT_SUMMARIZE_MIN_MESSAGES = 4

# Cap on the summary's own length. Without this, a summary that itself keeps
# growing every fold defeats the purpose — it needs to stay small enough to
# be "free" in every future prompt.
CHAT_SUMMARY_MAX_TOKENS = 400

# ── Chat intents ─────────────────────────────────────────────────────────────
# Every user turn is routed to exactly one of these before the agent runs.
INTENT_CHAT           = "CHAT"            # answer conversationally, tools allowed
INTENT_REPORT         = "REPORT"          # user wants the full document pipeline
INTENT_CONFIRM_REPORT = "CONFIRM_REPORT"  # user said yes to a pending report offer
INTENT_SMALLTALK      = "SMALLTALK"       # greeting / meta / capability question

CHAT_INTENTS = [INTENT_CHAT, INTENT_REPORT, INTENT_CONFIRM_REPORT, INTENT_SMALLTALK]

# ── Evaluation ────────────────────────────────────────────────────────────────
# The LLM-as-judge evaluator fires a background Groq call after every
# completed turn to score it against the rubric. That's genuinely useful
# signal, but it DOES consume Groq token budget per turn — which matters on a
# rate-limited free tier, especially during a live demo where you'd rather
# spend every token on answering the user than on grading.
#
# Default ON so the feature demonstrates itself out of the box. Flip to
# false in .env (EVAL_ENABLED=false) to spend ZERO evaluation budget — the
# api layer checks this flag before dispatching the scoring call, so "off"
# means the judge call never happens at all, not merely that its result is
# discarded.
import os as _os

EVAL_ENABLED = _os.getenv("EVAL_ENABLED", "true").strip().lower() not in (
    "false", "0", "no", "off",
)
