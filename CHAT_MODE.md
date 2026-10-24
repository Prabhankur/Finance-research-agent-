# Chat Mode

Conversational agent alongside the existing report pipeline. Same six tools,
same Groq client, different control flow and a FastAPI layer over both.

```bash
pip install -r requirements.txt
uvicorn api.server:app --reload --port 8000    # then open http://localhost:8000
python chat_cli.py                              # or talk to it in the terminal
python -m tests.test_chat_agent                 # 34 offline checks, no API keys
```

---

## Why a second graph instead of extending the first

The report pipeline plans everything up front (`roadmap_planner`), then
executes the plan. That's correct when you're producing a 25-section document:
the planning cost amortises over ~15 tool calls, and you want completeness.

Chat is the opposite problem. A user is watching a spinner and asked one
question. Planning ahead is wasted latency; you want to decide one step at a
time and stop as soon as the question is answered. So chat is a bounded ReAct
loop, not a planner.

What's shared is the part worth sharing — the tool layer (`tools/`), the Groq
client, the error taxonomy (`utils/error_classifier.py`), and the report-type
classifier. What's separate is the state shape and the control flow, because
forcing one graph to do both would mean every node carrying guards for fields
that are empty half the time.

---

## Architecture

```
                    ┌──────────────┐
                    │ intent_router│   rules first, LLM only if ambiguous
                    └──────┬───────┘
        ┌──────────────────┼───────────────┬──────────────┐
        │ agent            │ smalltalk     │ handoff      │ end
        ▼                  ▼               ▼              ▼
   ┌──────────┐       ┌─────────┐    ┌──────────┐        END
   │chat_agent│◄──┐   │smalltalk│    │ handoff  │   (report offer sent,
   └────┬─────┘   │   └────┬────┘    └────┬─────┘    awaiting yes/no)
        │ tools   │        ▼              ▼
        ▼         │       END            END
   ┌──────────┐   │
   │chat_tools├───┘   bounded by CHAT_MAX_TOOL_LOOPS
   └──────────┘
```

### New files

| File | What it does |
|---|---|
| `graph/chat_state.py` | `ChatState` — messages, intent, tool budget, tool trace, sticky ticker |
| `graph/chat_graph.py` | Compiled chat graph + checkpointer injection |
| `nodes/intent_router.py` | Routes each turn: CHAT / REPORT / CONFIRM_REPORT / SMALLTALK |
| `nodes/chat_agent.py` | Agent node, parallel tool node, smalltalk node, report handoff |
| `api/server.py` | FastAPI: SSE chat + background report jobs |
| `api/static/index.html` | Zero-build chat UI (streaming, tool chips, report card) |
| `chat_cli.py` | Terminal client, talks to the graph directly |
| `tests/test_chat_agent.py` | 34 offline control-flow checks |

`config.py` gained a chat section; `graph/__init__.py` became lazy (see below).

---

## Four design decisions worth defending in an interview

### 1. Rules before LLM in the router

Every turn needs an intent. Doing that with an LLM call costs latency and a
slice of the Groq TPM budget on *every* message, including "hi". So obvious
cases are caught by regex — greetings, explicit report requests, and any
message that names a company without a report keyword — and only genuinely
ambiguous turns reach the model. `intent_source` records which path fired, so
when routing misbehaves you know immediately whether to fix a regex or a
prompt.

### 2. The tool loop is bounded, and the bound is enforced by removing tools

`CHAT_MAX_TOOL_LOOPS` caps agent↔tool round trips. On the final round the
agent is re-invoked with **no tools bound at all**, so answering is the only
action physically available to it. That's stronger than prompting "please
answer now" — the model cannot ignore an instruction that doesn't exist in its
action space. The graph's conditional edge checks the budget too, as a second
line of defence against a future edit regressing the node.

Without this, an unbounded ReAct loop on a free tier (Alpha Vantage: 25
calls/day) can burn the entire daily budget on one bad turn.

### 3. Reports need confirmation, and they run as background jobs

Two separate decisions, both about the same thing: a report run is a multi-
minute, ~15-API-call commitment.

**Confirmation** — the router never launches a report. It stages the request in
`pending_report_*` and replies with an offer. Only an explicit "yes" on the
next turn sets `report_confirmed`. This is the human-in-the-loop checkpoint
from the roadmap, implemented at the routing layer rather than as a LangGraph
`interrupt()`. Same guarantee, and it survives a stateless HTTP boundary,
which a mid-graph interrupt does not without extra plumbing.

**Background jobs** — the graph emits `report_confirmed`, and the API dispatches
the pipeline to a worker thread with a polled status endpoint. Running it
inside the request would pin a worker for minutes and lose the whole run if the
client blinked. The graph owns the conversation; the API owns long-running
work.

### 4. Tool output is trimmed, and history windowing never orphans a tool message

A 10-K is roughly 500k characters. Pasted raw into the message history it
exceeds the context window in one call, so every tool result is trimmed to
`CHAT_TOOL_RESULT_MAX_CHARS` before re-entering the conversation. (The report
pipeline keeps the full text — it has a context router to shard it. Chat
doesn't.)

History windowing is subtler. Providers validate that every `ToolMessage`'s
`tool_call_id` matches a `tool_call` on a preceding `AIMessage`. A naive
`messages[-24:]` slice breaks that roughly whenever a turn used tools, and the
resulting 400 is opaque. `_window_messages()` drops orphaned pairs whole.
There's a test for it.

---

## Bugs the tests caught during the build

Written down because "I wrote tests and they found things" is a better story
than "I wrote tests".

1. **`P/E` became the active ticker.** Ticker extraction accepted 1-char
   uppercase tokens, so "what is Apple's P/E?" extracted `P`, which then stuck
   as the conversation's company for every following turn. Every subsequent
   lookup silently queried the wrong symbol — no error, just wrong data
   presented confidently. Fixed by requiring 2-5 chars and adding lookarounds
   that refuse to match inside `P/E`, `S&P`, `U.S.`.

2. **Dict order decided which company you meant.** "Compare TCS and Infosys"
   returned `INFY` purely because `"infosys"` sat earlier in the name→ticker
   dict literal. Now the earliest match *in the text* wins, across both
   symbols and names.

3. **A tool failure vanished from the trace.** Surfaced as an empty
   `tool_trace` in a test, traced back to the message being routed through the
   LLM intent path when a rule should have caught it. Fixed by the
   ticker-bearing-message rule in the router, which also removed an LLM call
   from the most common turn type.

---

## `graph/__init__.py` is now lazy

It used to eagerly import `financial_agent_graph`, which meant
`from graph.chat_state import ChatState` transitively imported every report
node and therefore yfinance, matplotlib, python-docx and edgartools.

Two real costs. Server startup paid seconds of import time for a pipeline most
requests never touch. And a missing optional dependency in one report-only tool
would break the chat agent *at import time*, even though chat never calls that
tool — a failure with no logical connection to its cause.

Compiled graphs are now exposed through module `__getattr__`. `from graph
import financial_agent` still works identically; the import just happens on
first access.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | One chat turn, streamed as SSE |
| `GET` | `/api/threads/{id}/history` | Stored transcript, for UI reload |
| `DELETE` | `/api/threads/{id}` | Returns a fresh thread id |
| `POST` | `/api/report` | Dispatch a report run → `job_id` |
| `GET` | `/api/report/{job_id}` | Poll status |
| `GET` | `/api/report/{job_id}/markdown` | Report as markdown |
| `GET` | `/api/report/{job_id}/download` | `.docx` (falls back to `.md`) |
| `GET` | `/api/health` | Liveness **and** which API keys are present |

### SSE events on `/api/chat`

| Event | Payload |
|---|---|
| `start` | `{thread_id}` |
| `status` | `{node, label}` — drives the "Fetching data…" indicator |
| `tool_calls` | `{calls: [name]}` — what the agent decided to call |
| `tool` | `{tool, status, error_class, duration_s, chars, truncated}` |
| `token` | `{text}` — answer chunk |
| `report_requested` | `{query, report_type}` — client POSTs `/api/report` |
| `done` | `{thread_id, answer, tools_used}` |
| `error` | `{message}` |

**SSE, not WebSockets:** the chat stream is strictly one-directional
server→client for the duration of a turn — the user's input already arrived in
the POST body. SSE gives that over plain HTTP with no upgrade handshake, no
keepalive logic, and no reconnection protocol to hand-roll. WebSockets become
the right call when the client needs to interrupt a running turn, which is
worth revisiting when cancel-mid-generation gets built.

---

## Known limitations

- **`token` events are chunked, not true token streaming.** LangGraph's
  `messages` stream mode requires the node to call `.stream()` rather than
  `.invoke()`, and `.invoke()` is what makes the 429 retry wrapper work
  reliably. Perceived latency is identical. Swapping to real token streaming is
  a contained change inside `chat_agent_node`.
- **`MemorySaver` loses conversations on restart.** `build_chat_graph()` takes
  a checkpointer argument, so `SqliteSaver`/`PostgresSaver` is a one-line change
  at the call site with no graph edits.
- **History is windowed, not summarised.** Past
  `CHAT_HISTORY_MAX_MESSAGES = 24`, the oldest turns are dropped rather than
  compressed.
- **Report jobs live in a process-local dict.** Fine for one uvicorn worker;
  needs Redis or a DB before scaling out.
- **The name→ticker map is ~35 entries**, not a security master. Unknown names
  fall through and the agent resolves them with a tool call.

---

## Still open from the previous session

Untouched by this work, still worth doing in the stated order:

1. `abs(capex)` before the free-cash-flow calculator (1 line)
2. yfinance `stock.info` fields for Company Profile (employees, HQ, website)
3. **Canonical financial data schema** — the root fix for calculator skips and
   N/A padding. Chat mode benefits from this too: the agent currently reads
   numbers out of text just like the compute executor does.
4. Caching for SEC filings and annual financials
5. Evaluation benchmark — with the chat graph in place, the same benchmark can
   now score both modes, which makes it a stronger claim

---

# Reliability & Knowledge Layer

Added after a run of production failures (Groq 404s, `tool_use_failed` 400s,
daily-quota exhaustion) destroyed data that successful API calls had already
fetched.

## 1. Partial-result preservation — the headline fix

Previously, if three tools succeeded and the *final synthesis* LLM call then
failed, the whole turn returned a bare "I hit an error" and silently threw
away every fetched figure — data that cost real quota against a finite daily
limit.

Now every successful tool result is preserved untrimmed in
`successful_tool_results`, and on synthesis failure
`utils/result_formatter.build_partial_response` renders it **in pure Python**
— so it cannot fail the same way the LLM call just did.

The output is explicitly framed as unfinished:

```
### Partial result — data retrieved, analysis incomplete
**These are raw retrieved figures, not a finished analysis**
**financial_data_api** (ticker=MSFT)
- Total Revenue: 331839000000
**Not completed:** cross-source interpretation, derived metrics...
**Technical detail:** `Error code: 400 - tool_use_failed`
```

Presenting raw figures as though they were finished analysis would be worse
than useless — reconciliation and interpretation are exactly what failed.

**Four turn states** replace the old success/error binary:
`COMPLETE` · `RETRIEVAL_PARTIAL` · `RETRIEVAL_COMPLETE_SYNTHESIS_FAILED` ·
`RETRIEVAL_FAILED`. Synthesis can be retried for free; retrieval cannot —
that distinction now exists in the data model.

## 2. `tool_choice="none"` on the final round

The final round used to *omit* tools to stop the loop, relying on Groq's
documented default (`tool_choice` defaults to `none` when no tools are
present). gpt-oss models emitted tool calls anyway — including their **own**
native `web_open`, never registered here — and Groq 400'd the whole request.

Tools now stay bound with an explicit `tool_choice="none"`, plus a recovery
path that catches `tool_use_failed` and retries once with tools stripped.
This deliberately does *not* trigger model fallback: it's a request-shape
problem, so swapping models would just burn the candidate list.

## 3. Live model catalog

`list_available_models()` calls `GET /v1/models` — ground truth for what the
current key can use — and ranks results by preference. Hardcoded names are
now a last-resort fallback, not the primary mechanism. Hardcoding was how
this project ended up calling a 404'd model repeatedly.

## 4. Context sized to the TPM limit, not the context window

A production 413 read: `TPM Limit 8000, Requested 10103`. On a free tier the
**tokens-per-minute cap binds long before any model's context window**, and
the whole request — system prompt + history + bound tool schemas — counts
against it. Budget is now 3500 tokens with 2500-char tool results, leaving
room for ~800 tokens of system prompt, ~1200 of tool schemas, and the
completion. Full payloads are still preserved for the *user* (see §1); the
cap only governs what re-enters the LLM's context.

## 5. Knowledge layer (Tier 1)

`database/` — runs on **SQLite or PostgreSQL**, chosen by `DATABASE_URL`
alone, no code change:

```bash
# default: zero-setup SQLite
export DATABASE_URL=postgresql://user:pass@localhost/findb   # switch to PG
```

Tables: `sources`, `companies`, `income_statements`, `balance_sheets`,
`cash_flow_statements`, `market_prices`, `valuation_snapshots`,
`macro_series`, `macro_observations`, `response_cache`.

**Caching** is wired into the `_fn` wrappers in `tools/__init__.py` — the one
chokepoint **both** the chat agent and the report pipeline already funnel
through, so a single wiring point covers the project and TTL policy lives in
one file. Measured: 5 requests → 2 live API calls.

TTLs answer "how long until this could legitimately change?", not "how stale
can we tolerate" — hence five orders of magnitude between them:

| Data | TTL | Why |
|---|---|---|
| SEC filings | 30d | immutable once filed |
| Earnings transcripts | 30d | immutable once published; AV is 25 calls/**day** |
| Annual financials | 7d | new data ~4x/year |
| Company profile | 14d | sector/employees move slowly |
| Quarterly / macro | 1d | quarterly cadence; FRED publishes daily |
| Valuation / prices / news | 1h | move with the market |

**Provenance:** every fact points at a `sources` row (provider, URL,
`retrieved_at`). You cannot flag "SEC says $391.0B but yfinance says
$391.04B" until both are stored as separately-attributed facts rather than
collapsed into one blob — this is the groundwork for evidence grading and
contradiction detection.

**Failure posture:** a broken DB degrades to *no caching*, never to *no
agent*. Every read path treats a cache error and a cache miss identically.
There's a test for this.

## Bugs the tests caught during this work

- **`operator.add` reducers cannot be reset.** Returning `[]` appends nothing
  and leaves prior contents intact. Without a fix, a synthesis failure on
  turn 10 would replay tool results from turns 1–9 as though they answered
  the new question. Fixed with a `reset_or_append` reducer.
- **`cached_call` itself was unguarded.** `get_cached`/`set_cached` were
  individually guarded, but the branching logic around them wasn't — a cache
  failure could still take down a tool call that should have just gone live.
- **Duplicate `CHAT_TOOL_RESULT_MAX_CHARS`** in `config.py` — the later
  definition silently won.

---

# Knowledge Layer — real database, real provenance

The previous "cache layer" section described a text-blob cache with a
`sources` table that was never actually used. This section replaces that
picture: the knowledge layer now normalizes provider responses into typed
rows in `income_statements` / `balance_sheets` / `cash_flow_statements` /
`macro_observations`, every fact linked to a real `sources` row.

## Why Postgres (for the project defense)

The schema runs on both SQLite and Postgres via one `DATABASE_URL` env var,
no code change. That said, the defensible reasons for Postgres are:

1. **Concurrent writes without global serialization.** Chat and the report
   pipeline can hit the same tables simultaneously; SQLite serializes ALL
   writers globally (WAL only helps readers), Postgres uses MVCC so
   concurrent writes to different rows don't fight.
2. **Real UNIQUE constraint enforcement across concurrent transactions.**
   Critical for "one row per `(company, period_end, source)`" without race
   conditions.
3. **Structured typed columns with real indexes** — a range scan for "5
   years of NVDA quarterly income statements" is O(log n), not the full
   scan a JSON-in-a-BLOB approach would require.
4. **pgvector without switching databases** — the semantic-search work
   over SEC filing chunks lives in the same DB.
5. **Foreign-key provenance enforced at the DB layer**, not hoped-for at
   the app layer.

What I don't claim: Postgres is faster than SQLite at our current scale.
It isn't, for single-process reads. The value is correctness under
concurrency and future-proofing for pgvector — not raw speed.

## The three-layer read path

```
financial_data_api_fn(ticker=MSFT, statement_type='income', years=2)
    │
    ▼
1. try_rebuild_financials()   <-- NEW
    │  ├── DB has fresh income rows for MSFT annual, 2 periods?
    │  ├── YES: rebuild the same text response, RETURN. No API. No cache read.
    │  └── NO:  fall through
    ▼
2. cached_call() → get_cached()
    │  ├── Exact same call args in response_cache, unexpired?
    │  ├── YES: RETURN payload.
    │  └── NO:  live fetch
    ▼
3. live yfinance/AV
    │  ├── record_source(provider='yfinance') → source_id
    │  ├── set_cached(payload)
    │  ├── normalize_yfinance_text(payload, source_id) → typed rows
    │  └── RETURN payload
```

Layer 1 is what turns different-shaped requests into cache hits:
`(statement='all', years=3)` and `(statement='income', years=2)` are
different text-cache keys, but they read from the same underlying rows.
Measured: **1 API call across 3 differently-shaped requests** in the E2E
test — the report pipeline no longer double-fetches what chat already got.

## Provenance is now real

Every normalized row has `source_id → sources.id`. Sources rows record:
- `provider` — yfinance / alpha_vantage / fred / sec / tavily
- `retrieved_at` — exact timestamp
- `url` — where applicable

This means "as of 2026-08-31" queries work, and the future contradiction
detector has something to check against: the same `(company, period_end,
revenue)` from two providers lands as two rows with two `source_id`s
rather than one silently overwriting the other.

## Report jobs are now persistent

Reports used to live in a Python dict wiped on every server restart —
completed .docx files sat orphaned on disk with no API memory of them.
Now backed by `report_jobs` table (created in `database/report_jobs.py`)
with a `thread_id` column, so:
- Reports survive restarts
- `GET /api/threads/{id}/reports` lists reports from a specific conversation
- The sidebar can show "reports from this chat" (frontend surface TBD)

## Two UI bugs fixed

- **"New chat" during streaming used to orphan the response.** No
  AbortController meant the fetch reader kept writing tokens into deleted
  DOM nodes, `busy` stayed latched, and the backend kept generating (still
  billing Groq tokens) for a response nobody would see. Fixed with a real
  AbortController wired into New Chat and thread-switch.

## What's still deferred (honestly)

- SEC filing text normalization — the XBRL parse is fragile; the raw text
  cache still covers it, provenance not yet recorded.
- News and transcript normalization — same story.
- Contradiction detector node — the DB structure supports it now; the
  actual node is Phase B, not yet implemented.
- Evidence grading (A/B/C/D source tiers) — Phase B.
- Self-evaluator (LangSmith rubric scoring) — Phase B, and correctly
  sequenced AFTER contradiction detection so it has real behaviors to
  score against.

---

# Phase C — Token/cost optimization (honestly measured)

Phase C targets the failure mode observed in production: a 3-company
comparison would hit Groq's `413 Request too large` error at ~10000 tokens
against the 8000 TPM cap. Concretely three changes:

## 1. `fields` parameter on `financial_data_api`

Response-shaping (not scope-scoping): the fetch, cache, and DB all still
hold the full response, but the model can ask for only the metrics it
needs. Recognized aliases: revenue, net_income, gross_profit, gross_margin,
operating_income, operating_margin, net_margin, eps, pe, market_cap, cash,
debt, assets, equity, roe, cash_flow, free_cash_flow, price, dividend, beta.

Failure mode is safe by design — unknown field name falls back to the full
response rather than dropping data the model needed.

## 2. Calculator docstring compression

The calculator's 27-parameter schema was 1171 tokens. Compressed the
per-metric field-list docstring while keeping identical guidance. Saved
~73 tokens per bound-tool call.

## 3. System prompt audit

Compressed 911 → 640 tokens (30% reduction) without dropping any behavior:
- Merged redundant "answer in chat, not a document" phrasing
- Tightened source-tier and tool-usage sections
- Kept all hard rules on numbers and the citation format
- Added explicit `fields=[]` guidance so the model actually uses it

## Measured impact — the honest number

Isolated per-call overhead (system prompt + tool schemas): 4058 → 3909
tokens (only 3% — the `fields` parameter cost 196 tokens, partially
offsetting the system prompt savings).

**The real win is on multi-company comparison queries:**

| Scenario | Before | After | Δ |
|---|---|---|---|
| 3-company comparison, tool round | 5784 tokens | 4170 tokens | **-28%** |
| Synthesis round (includes prior results in history) | 6284 tokens | 4670 tokens | **-26%** |
| Room under 8000 TPM cap | 1716 tokens | 3330 tokens | **~2×** |

The 8000 TPM `413` errors were happening at 6000-10000 tokens on those
comparisons. Post-Phase C, the same query with `fields` used lands under
5000 tokens for both rounds — meaningfully clear of the limit.

**Important honest caveat:** these savings depend on the model actually
using the `fields` parameter. The system prompt now explicitly guides
this, and LangSmith traces will tell you within a day whether the model
picks it up in practice. If it doesn't, the fallback is the fields param
is still valuable when tools called directly (e.g. from the report
pipeline), which we control.

## Bug the tests caught

The initial field-filter regex was `f"{label}:" in stripped` — needed the
colon immediately after the label. But yfinance's statement format pads
whitespace before the colon: `Total Revenue      :  245122000000`. The
naive check matched nothing on statement responses (only overview JSON),
silently returning empty. Fixed with a whitespace-normalization pass
before matching; test proves both formats work.

---

# Final completion pass — all four remaining items

## 1. Postgres — actually verified against a LIVE instance, not just schema-compatible in theory

Installed Postgres 16 in a real sandbox and ran the full functional suite
against it: caching, SEC/yfinance normalization, DB-first reads, and
contradiction detection all confirmed working end-to-end on a real server,
not SQLite.

**This surfaced a real, serious bug that was invisible on SQLite:**
`psycopg2` cursors have a `.lastrowid` *attribute* (so the old
`getattr(cursor, "lastrowid", 0)` never fell through to its default), but
Postgres has no session-level "last insert id" the way SQLite does — it
always reads back `None`/`0`. Every `execute()` call was silently reporting
id=0 for a row that had actually been inserted correctly with a real id.
That 0 then got used as a foreign key elsewhere (e.g. `sources.id` passed
into `income_statements.source_id`) and caused a `ForeignKeyViolation`
several calls later — a confusing failure surfacing far from its actual
cause.

**Fix:** on Postgres, INSERT statements get `RETURNING id` appended
automatically and the real id is read from the query result. Falls back
to a plain insert (returning 0) for tables with no `id` column at all —
`report_jobs` is keyed by `job_id` (TEXT), not `id`, and its caller never
uses the return value anyway.

Verified via a live test (skips gracefully if no Postgres is reachable, so
it won't break a SQLite-only environment): `execute()` now returns the
row's actual id, confirmed to match what's really in the table.

## 2. Contradiction UI banner

`api/static/index.html` now handles the `contradiction` SSE event with a
visible amber banner: which sources disagreed, their values and tiers, and
which one the answer led with. Previously this data only reached the
model's prompt — now the user sees the same disagreement directly.

## 3. SEC / news / transcript provenance and normalization

**SEC filings** — full structured normalization, not just caching. A
dedicated parser (`normalize_sec_xbrl_text`) handles SEC's XBRL text
format, which is structurally different from yfinance's: scaled
human-readable values (`$391.04B`) with the period embedded inline per
line, rather than raw integers under a shared `Period:` header. Since SEC
is tier A, this means the contradiction detector can now compare a real
SEC figure against a yfinance/AV figure for the same fact — exactly the
comparison this project exists to make possible.

**News and transcripts** — deliberately scoped to **provenance only**, not
full structured parsing. This was a conscious decision, not a shortcut:
there's no numeric fact to extract from a news snippet's prose, and
contradiction detection only operates on numeric facts anyway — a
"news_articles" table with parsed rows wouldn't earn its complexity. What
does matter and is now real: a `sources` row gets recorded for every
`web_search` and `earnings_transcript` call (provider, retrieved_at),
closing the "when did we search for this" gap without pretending
unstructured prose can be normalized like a balance sheet.

This required loosening `cached_call`'s provenance gating — it previously
required BOTH a `normalize` callback AND a `provider` to record anything,
which meant unstructured tools got no provenance at all. Now `provider`
alone is sufficient.

## 4. Calculation-fidelity — deterministic ground truth for calculated metrics

Directly answers the "the evaluator should at least know the correct
answer" concern. For CALCULATED metrics specifically, there IS a ground
truth: the calculator is pure deterministic Python. So this is not an LLM
judgment — it's a Python string/number match between what the calculator
actually returned this turn and what number appears in the final written
answer.

New 6th rubric dimension: `calculation_fidelity_score`. `NULL` when no
calculator call happened this turn (not applicable — same convention as
`reconciliation_score` with no contradictions found), `100` when every
calculator result is reflected in the answer, scaled down per mismatch
otherwise. Mismatches are logged with the calculator's args and actual
result for debugging.

Also added: `scores_for_thread(thread_id)` and
`GET /api/threads/{id}/evaluation` — the `thread_id` column has existed on
every scored turn since the evaluator was built, but there was no query
endpoint filtering by it until now.

## All 171 tests pass, including a real live-Postgres integration test

The Postgres test connects to an actual server and runs real INSERT/SELECT
statements — not mocked. It skips cleanly (printing why) if no Postgres is
reachable, so it won't break CI or a developer's SQLite-only machine.
