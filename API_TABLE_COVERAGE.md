# API ↔ Table Coverage Audit

**Method.** For every tool: what it accepts, what the upstream API returns, what
the tool actually *prints*, what the normalizer parses, and which column the
value lands in. A table can only hold what the tool prints, and the tool can
only print what the API call asked for — so gaps show up at one of three layers.

Status: ✅ stored · 🆕 stored as of this version · ⚪ raw cache only (by design) · ❌ gap

---

## 1. `financial_data_api` — yfinance (primary), Alpha Vantage (fallback)

**Accepts:** `ticker`, `statement_type` (`overview` | `income` | `balance` | `cashflow` | `ratios` | `all`), `period` (`annual` | `quarterly`), `years` (1–5)

### Overview block (yfinance `.info`, printed as JSON)

| Printed field | Destination | Status |
|---|---|---|
| Company Name, Sector, Industry, Currency | `companies.name / sector / industry / currency` | 🆕 (was NULL for every ticker) |
| Market Cap, Trailing P/E, Forward P/E, EV/EBITDA, Price/Book, Beta | `valuation_snapshots` | 🆕 |
| Current Price, 52-Week High/Low, 50/200-Day MA | `valuation_snapshots.price / week52_* / ma_*` | 🆕 new columns |
| Profit Margin, ROE, Debt/Equity, Revenue TTM, Net Income TTM | `valuation_snapshots` | 🆕 new columns |
| Dividend Yield (%) | `valuation_snapshots.dividend_yield` | 🆕 and the value itself was corrected (see below) |
| Business Summary | — | ⚪ |

### Statements (yfinance DataFrames)

| Statement | Printed | Stored | Columns that exist but yfinance path never fills |
|---|---|---|---|
| Income | Revenue, Gross Profit, Operating Income, Net Income, EPS (Basic), EBITDA, R&D | ✅ all 7 | `cost_of_revenue`, `eps_diluted`, `currency` |
| Balance | Total Assets, Total Liabilities, Equity, Cash, Total Debt, Retained Earnings | ✅ all 6 | `current_assets`, `current_liabilities`, `currency` |
| Cash flow | Operating CF, CapEx, Free CF, Dividends Paid, Share Buybacks | ✅ OCF/CapEx/FCF, 🆕 Dividends + Buybacks | `investing_cash_flow`, `financing_cash_flow`, `currency` |
| Ratios (11 fields) | PE, PEG, Book Value, EPS, ROE, ROA, margins, growth, analyst target | ⚪ | not normalized |

---

## 2. `fred_series` — Federal Reserve Economic Data

**Accepts:** `series_id` (code or shortcut such as `gdp`), `observation_start`, `observation_end`, `limit`

| Endpoint | Returns | Destination | Status |
|---|---|---|---|
| `/series/observations` | date, value | `macro_observations` | ✅ (was **empty** — regex rejected comma-formatted values) |
| `/series` | title, units, frequency, seasonal adjustment | `macro_series.name / unit / frequency / seasonal_adjustment` | 🆕 (tool never called this endpoint before) |

Also fixed: shortcuts (`gdp`) are resolved to the real id (`GDP`) before storing,
so one series can no longer be registered twice.

---

## 3. `sec_filing_search` — SEC EDGAR XBRL company facts

**Accepts:** `ticker`, `filing_type` (`10-K` | `10-Q`), `year`

| XBRL concept (label) | Destination | Status |
|---|---|---|
| Revenue, Net Income, EPS Diluted, Gross Profit, Operating Income, R&D | `income_statements` | ✅ |
| Cash, Shareholders Equity | `balance_sheets` | ✅ |
| Long-Term Debt | `balance_sheets.long_term_debt` | 🆕 (was written into `total_debt` — a different measure; the contradiction detector then compared it against yfinance's *Total* Debt) |

All 9 fetched concepts are mapped. Existing SEC rows are relocated automatically.

---

## 4. `web_search` (Tavily), `earnings_transcript` (Alpha Vantage)

| Tool | Destination | Status |
|---|---|---|
| web_search | `response_cache` + `sources` | ⚪ prose — no numeric fact to normalize |
| earnings_transcript | `response_cache` + `sources` | ⚪ prose — future home is a vector index, not columns |

## 5. `calculator`

Pure local arithmetic. No API, no storage.

---

## Table status after this version

| Table | Writer | Notes |
|---|---|---|
| companies | ✅ | metadata now populated |
| income_statements / balance_sheets / cash_flow_statements | ✅ | see gaps below |
| valuation_snapshots | 🆕 | **had no writer that ever ran** (only INSERT lived in `normalize_av_overview`, which nothing calls) |
| macro_series / macro_observations | ✅ / ✅ | observations were empty until the regex fix |
| market_prices | ❌ | no tool fetches OHLCV history |
| sources, response_cache, report_jobs, evaluation_scores | ✅ | |

---

## Currency (added)

yfinance returns each company **in its own currency and never converts**
(`MSFT` → USD, `TCS.NS` → INR, raw rupees not crores). Two different facts exist:

| Fact | yfinance field | Stored in |
|---|---|---|
| Currency the share trades in | `info["currency"]` | `companies.currency` |
| Currency the statements are reported in | `info["financialCurrency"]` | `companies.financial_currency` + `currency` on every statement row |

They differ for ADRs (trades in USD, reports in INR), so they are stored separately.
Statement headers now read `Income Statement (Annual) – TCS.NS [reporting currency: INR]`;
Alpha Vantage statements use their own `reportedCurrency`; SEC rows are `USD` (the tool
reads only USD / USD-per-share units).

**Market suffix.** A bare ticker means the US listing. The `financial_data_api` ticker
description and the system prompt now teach `.NS` (NSE), `.BO` (BSE), `.L`, `.TO`.

**Conversion.** `fx_rate(from, to)` returns a market reference rate (yfinance `USDINR=X`,
falling back to the inverse pair). It is chat-only, used **only when the user asks to
convert**, and the model multiplies with `calculator` so arithmetic stays in one place.
Cached 1 hour; error payloads are never cached. The prompt forbids adding, comparing or
ranking figures across currencies. The report pipeline does not get this tool.

Cost: about +270 tokens on every chat call (151 for the tool schema). Remove `"fx_rate"`
from `CHAT_ENABLED_TOOLS` in `config.py` to save the 151 if conversion is not needed.

---

## Read path for `financial_data_api` (corrected)

| Order | Layer | Serves | Why it sits here |
|---|---|---|---|
| 1 | **Exact-match text cache** (`response_cache`) | An identical repeat | One indexed lookup; holds the *original* response, including the Market Overview and Key Ratios the typed tables cannot rebuild. Never worse than a rebuild. |
| 2 | **Knowledge layer** (typed tables, `try_rebuild_financials`) | A *differently-shaped* request for data already stored (`years=2` after `years=3`) | The only layer that can answer a request whose cache key was never seen. Single statements only. |
| 3 | **Live provider** | Everything else | Records provenance, normalizes, stores. |

Previously the order was 2 → 1 → 3. That was wrong in four ways, all found by running the layers against each other:

- an identical repeat got the thinner rebuilt text although the full original was cached;
- the rebuild **dropped the reporting-currency tag and Dividends Paid / Share Buybacks**;
- `statement_type="all"` (the tool's default) was rebuilt from three of its five blocks, **silently dropping the Market Overview and Key Ratios**;
- once any period had been fetched twice, it returned **one fiscal year twice and omitted the other**.

## Row duplication on refetch (fixed)

`record_source()` writes a new `sources` row per retrieval and fact tables are unique on
`(company, period_end, period_type, source_id)`, so a real refetch never hit `ON CONFLICT`:
it inserted a second copy of every period. Now the newest fetch *replaces* older rows from the
same provider (SEC and yfinance rows still coexist — that pair is what contradiction detection
compares), reads keep one row per period, and an idempotent migration removes legacy duplicates.

## Worth checking against real data (not changed)

The contradiction detector matches periods by **exact date equality** (`fs.period_end = ?`).
SEC reports the actual fiscal year-end (e.g. 2024-09-28); yfinance may label the same year
differently (e.g. 2024-09-30). If so, those two rows never meet and the conflict is never seen.
To check, after fetching both for one company:

```sql
SELECT s.provider, i.period_end, i.revenue
FROM income_statements i JOIN sources s ON s.id = i.source_id
JOIN companies c ON c.id = i.company_id
WHERE c.ticker = 'AAPL' ORDER BY i.period_end DESC;
```

---

## Failure policy (when a tool fails: retry in code, or reason?)

Rule: **retry in code what is mechanical; bring in the model only where a decision is needed.**
A code retry costs one tool call. A model-driven retry costs a whole round of the 4-round
budget plus an LLM call (~5–6k prompt tokens in production traces).

| Failure type | Example | Chat | Report (fetch step) |
|---|---|---|---|
| **Transient** | timeout, dropped connection, 5xx | one retry in code after `TRANSIENT_RETRY_PAUSE_S` (1.5 s); no model call, no extra round | same (previously retried with a 0.00 s gap) |
| **Bad args** | `years=0`, unexpected keyword | not retried in code; model is shown the exact error; an *identical* repeat is refused unexecuted | one model call repairs the arguments, one retry (previously re-sent the SAME args) |
| **Systemic** | invalid/missing key, persistent rate limit | never retried; tool withdrawn for the rest of the turn (not bound, refused if requested); resets next message | never retried; tool disabled + replanner (unchanged) |
| **Legitimate empty** | no filings found | returned with "valid result — do not retry" | tagged "not available" (unchanged) |

Also fixed while doing this:

| # | Bug | Effect |
|---|---|---|
| 13 | Classifier searched failure keywords across the **whole** result | a successful search about "unauthorized trading" was labelled `systemic` — once that disables a tool, a working tool would be withdrawn. Failure keywords are now trusted only on error-shaped text. |
| 14 | Chat labelled every **raised exception** `systemic` | a plain `TypeError` (bad arguments) or dropped connection looked like a permanently broken tool. Now classified from the message. |
| 15 | Report fetch retried bad arguments with the same arguments | cannot succeed by construction. |

Limits, stated plainly: the compute step's one-line pause (LLM-call transient errors) is not
separately tested; withdrawn-tool behaviour with the live Groq model is untested (the tool is
simply not offered, so it should hold, but watch the first real failures); the error-shape
threshold (400 characters) is a heuristic.

---

## Concurrency, rate limits, report input, cost (latest changes)

| Change | Detail | Evidence |
|---|---|---|
| **Request coalescing (single-flight)** in `cached_call` | first request for a key leads; identical concurrent ones wait for its result; double-checked after winning leadership; a failing leader releases waiters; waiters time out after 90 s; an error text is shared, not re-requested | 8 simultaneous identical requests: **8 live calls before, 1 after**, 0.32 s vs 0.31 s. Process-local — several worker processes would need a shared lock |
| **429 handling** | no sleep after the final failed attempt; a *daily* (TPD) 429 marks that model exhausted for 15 min and falls back to the next model; a per-minute 429 is still waited out | **Not verified live:** assumes Groq quotas are per model |
| **Report writer receives the user's query** | `raw_query` (≤600 chars) is now in every section prompt; it was never passed before | test: query text present in the section prompt |
| **Fair truncation** | shared `utils/text_budget.fair_share` used by the rescue prompt and the evaluation judge; both previously cut later results hardest | 5 equal results now keep equal shares; the last gets as much as the first |
| **System prompt tightened** | 982 → 885 tokens (Phase C had reached 640; real capabilities added since) | measured with the project's estimator |

### What the model receives (measured, 3-turn run through the real graph)

| Call | system | human | ai | raw tool data | total |
|---|---|---|---|---|---|
| 1 small talk | 159 | 1 | 0 | 0 | 160 |
| 2 turn 2, deciding | 885 | 9 | 9 | 0 | 903 |
| 3 turn 2, after tool | 885 | 9 | 10 | 480 | 1,384 |
| 4 turn 3, follow-up | 885 | 16 | 25 | 480 | 1,406 |

Fixed overhead per chat call ≈ **5,235 tokens** (system 885 + tool schemas 4,350; `calculator` alone is 1,538).
Earlier turns' **raw tool results stay in the window** until it overflows. The evaluation judge sees the question, the answer,
**this turn's** tool results (≤2,000 chars) and contradictions (~1,400 tokens) — not the summary, not earlier turns.
The report writer sees ≈2,300 tokens per section and neither other sections nor chat history.

Cost lever not taken: a compact `calculator` argument schema (would cut several hundred tokens per call) — held back because
a schema change affects tool-call reliability, which cannot be validated offline.

---

## Question understanding, the batch tool, entities in state (latest)

| Change | Detail | Evidence |
|---|---|---|
| **Batch `tickers` parameter** on `financial_data_api` | one call carries up to 8 companies; the chat tool node expands it into one parallel fetch per company, each with its own retry / withdraw / trace / preserved result; the model gets ONE merged reply (header per company, summary line); companies share one context budget; overflow and non-ticker names are listed as "Not fetched" | one call naming 8 companies: **1 round, 8 fetches, 0.5 s**; the real run used 4 rounds for 4 of 8 |
| **Understanding step ("front door")** | one small structured call writes a *brief* (scope, subjects, metrics, period, assumptions, optional question) beside the user's UNCHANGED message; the model proposes, the code decides: a question is asked only when no subject can be found anywhere; a missing year becomes a stated assumption; the code's own name table beats the model's tickers; any malformed reply falls back to the rules-only router | `FRONT_DOOR_ENABLED` (default true); 21 new tests incl. end-to-end through the real graph |
| **Two new intents** | `CLARIFY` ends the turn with the question; `OUT_OF_SCOPE` takes the cheap no-tools path with a redirect | neither is scored by the evaluator (turn_status stays empty) |
| **State holds every subject** | `active_tickers` (list) and `brief`; `active_ticker` kept as the first entry. Follow-ups keep the company; a new topic clears it (a question about India's GDP used to arrive tagged with the previous company) | tested |
| **Prompt** | names all companies instead of "the company under discussion is AAPL"; carries the brief (~95 tokens) | tested |

Bugs found along the way: the ticker pattern allowed 12 characters but `TATAMOTORS.NS` / `BAJFINANCE.NS` are 13 (silently rejected) — fixed
to 20; `ticker` strings carrying several symbols and list items containing company *names* ("Apple Inc") were handled without silently
dropping or fabricating tickers.

**Measured cost.** `financial_data_api` schema +129 tokens on every agent call (847); understanding call ≈ 680 tokens and ~0.5–1 s per
substantive question; agent system prompt 841. On an 8-company question: ≈ 27,700 → ≈ 11,800 tokens (−57%), all companies returned.
On a simple single-company question it is a net *cost* (~700 tokens) — set `FRONT_DOOR_ENABLED=false` to use the rules-only router.

**Not verified live:** the understanding step has only run against a scripted model; how often gpt-oss returns valid JSON for it is unmeasured.
Parallel calls across *different* tools in one message still depend on the model.

---

## Python 3.9 compatibility, canonical cache keys, cache purge (v11)

| Change | Detail | Evidence |
|---|---|---|
| **Python 3.9 startup crash fixed** | `list[str] \| None` in `tools/__init__.py` raised `TypeError` at import on Python 3.9 (it needs 3.10+ unless `from __future__ import annotations` is present). Now `Optional[list[str]]`; the file deliberately does NOT get the future import because it holds the `@lc_tool` functions whose schemas LangChain builds from their annotations | reproduced on a real Python 3.9.25, then the whole suite (504 tests) and the FastAPI app (`/api/health` → 200) were run on 3.9.25 and 3.12 |
| **Permanent guard** | `test_the_code_base_has_no_syntax_newer_than_python_3_9` parses every source file with `feature_version=(3, 9)` and rejects `X \| Y` annotations / `isinstance` unions / `zip(strict=)` | runs in the normal suite |
| **Canonical cache keys** | `canonical_args`: ticker upper-cased and trimmed; `statement_type` / `period` lower-cased and trimmed; FRED shortcuts and padding resolved before the key is built. Entry points canonicalize too, so a lower-case spelling no longer creates a second company row | 4 spellings → 1 live call, 1 company row; `' gdp '` = `GDP` |
| **Cache purge is now called** | `purge_expired` runs from cache writes, at most once an hour; a failing purge cannot break a write | tested: runs once, not twice inside the hour, runs again after it |

Python versions: the project runs on 3.9 (what the author uses) and 3.12; both are exercised.

---

## Open gaps — each needs a decision, none is silently "done"

1. **Fields the tool never prints** (`cost_of_revenue`, `current_assets`, `current_liabilities`,
   `investing/financing_cash_flow`, `eps_diluted`). Printing them costs tokens on every call.
   Concrete consequence today: the calculator's `current_ratio` metric needs
   `current_assets`/`current_liabilities`, and no tool returns them.
2. **`market_prices`** — needs a new price-history tool (`yf.Ticker().history()`); new feature, not a fix.
3. **Alpha Vantage fallback is recorded as `yfinance`.** `cached_call(..., provider="yfinance")` is
   hard-coded, so when AV answers, provenance is wrong and cross-source comparison never sees AV.
4. **`normalize_av_*` (4 functions) have no caller.** Wire them or delete them.
5. **Overview fields never requested** though columns exist: `enterprise_value`, `price_to_sales`, `peg`.
6. ~~`currency` on the statement tables is never filled.~~ **Closed** — see *Currency* below.
7. **`macro_series.country`** — FRED's `/series` response has no country field.

---

## Bugs this audit found

| # | Bug | Effect | Fixed |
|---|---|---|---|
| 1 | FRED regex rejected `29,929.55` | `macro_observations` empty | ✅ |
| 2 | Overview JSON never parsed | `companies` metadata NULL, `valuation_snapshots` never written | ✅ |
| 3 | Dividends/buybacks printed, no column | silently discarded | ✅ |
| 4 | SEC LongTermDebt stored as `total_debt` | false cross-source contradictions | ✅ |
| 5 | `dividendYield × 100` | yfinance changed units; Apple showed **31%** | ✅ (`dividendRate ÷ price`) |
| 6 | FRED observations carry no units | agent could not tell SAAR from raw quarterly | ✅ |
| 7 | Shortcut ids stored as-is | duplicate series rows | ✅ |
| 8 | `CREATE TABLE IF NOT EXISTS` never alters tables | edits never reached existing DB files | ✅ migrations |
| 9 | Income upsert's `ON CONFLICT ... DO UPDATE` omitted `currency`, `eps_*`, `cost_of_revenue` | a row that already existed was never refreshed or back-filled | ✅ |
| 10 | Statements carried no currency | `TCS.NS` rupees looked like dollars; cross-currency comparison was silent | ✅ |

| 11 | Refetch inserted duplicate period rows (`source_id` in the unique key) | same fiscal year served twice, another omitted; table grew per refetch | ✅ |
| 12 | DB rebuild lossy, ran before the exact cache | thinner answer than the cached original; `all` silently partial | ✅ |
| 16 | Cache stampede — identical concurrent requests each hit the live API | 8 calls instead of 1 | ✅ single-flight |
| 17 | 429 loop slept after the last attempt; daily quota could not fall back to another model | wasted wait; needless failure | ✅ |
| 18 | Report writer never received the user's own query | scope words lost | ✅ |
| 19 | Judge input truncated later tool results hardest | correct figures could look ungrounded | ✅ |

## Reported, deliberately not changed (report pipeline)

`report_generator._write_section` builds each section prompt from the ticker, the section
instructions, the research data and a "roadmap notes" blob (the first 1,500 characters of
the outline JSON). **The user's own query is never included**, although the section
instructions refer to "the query". Wording such as "in rupees" or "FY2024 only" survives
only if the elaboration and roadmap steps happen to carry it.
