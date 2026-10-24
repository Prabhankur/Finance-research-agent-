# Test Plan — verifying each piece of the redesign

Run these with `python main.py --query "..." --stream` and, if LangSmith
tracing is on, cross-check the trace. For each test, the "what to check"
column is more important than whether the report looks good — a good-
looking report can still hide a silent failure underneath.

## 1. Baseline / happy path
```
python main.py --query "Analyse Apple AAPL for FY2024, including earnings, risk, and investment recommendation" --stream
```
**Tests:** the whole pipeline with nothing artificially broken.
**Check:**
- Console shows `fetch_executor` processing steps with `success` status, no retries.
- `compute_executor` shows calculator step(s) with real numbers in the result
  (e.g. `CAGR = 10.10%`), not a `TypeError`.
- No `[Replanner]` line appears at all.
- Report has no `[Data unavailable]` lines.

## 2. Calculator / compute_executor specifically
```
python main.py --query "Compute Apple's revenue CAGR over the last 3 years, its Return on Equity, and its Free Cash Flow Yield, with supporting financial data" --stream
```
**Tests:** forced `tool_choice`, and whether the model can actually locate the
right numbers in accumulated context and call calculator correctly for
*multiple different metrics* in the same run (each needs different named args).
**Check:** each compute step shows `status: success` in console; the actual
numeric results appear correctly in the report body, not `[Calculator]
Missing or wrong argument...`.

## 3. 10-Q XBRL fix — the original bug report
```
python main.py --query "Give me Apple's most recent 10-Q quarterly revenue, margins, and cash flow" --stream
```
**Tests:** the specific fix to `tools/sec_filing.py` — XBRL structured data
now pulls for 10-Q, not just 10-K.
**Check:** the SEC tool's returned content should contain a
`[SEC XBRL] Structured Quarterly Financial Data` block with real quarterly
figures, not just filing text with no numbers.

## 4. Legitimate empty result (not a failure)
```
python main.py --query "Analyse Tata Consultancy Services TCS.NS for FY2024" --stream
```
**Tests:** the `LEGITIMATE_EMPTY` classification path — a non-US ticker has
no SEC filings, and that's a valid answer, not an error.
**Check:** `sec_filing_search` step shows `status: legitimate_empty` (or
however you're printing it) — NOT `failed`, NOT a retry, NOT a replan. The
report should say SEC data isn't applicable for this ticker rather than
treating it as a data gap needing an apology.

## 5. Genuine mid-run systemic failure → replanner actually fires
This is the important one — don't just remove a key, **corrupt it** so
`check_env()` still sees it as present (so it does NOT get pre-seeded into
`disabled_tools` before the run starts) but the real API call fails:

```bash
# In your .env, temporarily set:
TAVILY_API_KEY=this_is_not_a_real_key_12345
```
```
python main.py --query "Analyse Apple AAPL for FY2024 including recent news and analyst sentiment" --stream
```
**Tests:** the actual failure → classify → escalate → replan path, live.
**Check console for, in order:**
1. `web_search` step fails with an auth-type error, classified `systemic`.
2. `[FetchExecutor]` stops the fetch pass (doesn't keep burning through the
   rest of the plan with a dead tool).
3. `[Replanner]` line fires, showing a revised remaining-steps list and a
   note like *"web search unavailable this run"*.
4. Run continues and completes — does not crash, does not hang.
5. Final report contains a `[Data unavailable]` note in whichever section
   declared `web_search` as a data source, worded as an actual explanation,
   not silently missing.

Revert the key afterward.

## 6. Exhausting the global replan budget
Repeat test 5 but corrupt **two** keys at once:
```bash
TAVILY_API_KEY=not_real_12345
ALPHA_VANTAGE_KEY=not_real_67890
```
```
python main.py --query "Analyse Apple AAPL for FY2024, comprehensive report" --stream
```
`config.GLOBAL_REPLAN_BUDGET = 2`, so this should use up the entire budget
(one replan per broken tool). To actually see the budget get *exceeded*
rather than just used up, also temporarily break a third source:
```bash
pip uninstall edgartools -y
```
**Check:** the third systemic failure should NOT trigger a `[Replanner]`
call — console should show it going straight to tagging the failure and
continuing (look for the "budget already exhausted" behavior in
`fetch_executor.py`), and the run still completes rather than looping.
Reinstall after: `pip install edgartools`.

## 7. Bad-args retry (should self-correct, no replan)
Hard to force deliberately since the calculator schema fix makes this rare
now — but if you see any `[Calculator] Missing or wrong argument` in a run,
confirm it self-corrects: the SAME step should show a second attempt in the
`compute_executor` log with `retries_used: 1` before either succeeding or
giving up — and it should NOT trigger a `[Replanner]` call, since bad-args
never escalates.

## 8. Missing key from the very start (contrast with test 5)
```bash
# Actually remove/comment out ALPHA_VANTAGE_KEY entirely
```
```
python main.py --query "Analyse a small AAPL earnings summary" --stream
```
**Tests:** the pre-seed path (soft warning), as distinct from test 5's
live-failure path.
**Check:** console shows the `⚠ Missing optional environment variables`
warning at startup, `earnings_transcript` is skipped with status `skipped`
from step 1 (never actually attempted), and — importantly — **no
`[Replanner]` call happens for this one**, since it was disabled before the
graph even started. This confirms the two paths (pre-seeded vs. live
failure) are behaving differently, as designed.

---

If you have LangSmith tracing on, the fastest way to confirm all of this is
to look at the trace tree for run node names — you should see
`fetch_executor`, `compute_executor`, `context_router`, and (only in tests
5 and 6) `replanner`. If you ever see `llm_reasoning`, `tool_node`,
`context_accumulator`, or `rag_retriever` appear, something old got left in
the loaded code — that would mean stale bytecode or a leftover file, not a
logic bug.
