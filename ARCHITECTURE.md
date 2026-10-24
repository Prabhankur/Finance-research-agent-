# Autonomous Financial Report Agent — Architecture

A LangGraph agent that takes a natural-language research query (e.g. "Analyse
Apple AAPL for FY2024"), plans a research strategy, gathers data from
multiple financial data sources, computes derived metrics, and writes a
structured investment research report — recovering from tool failures along
the way instead of silently producing a report with holes in it.

## Folder structure

```
financial-agent/
├── main.py                    # CLI entry point
├── config.py                  # single place for retry/replan budgets & tuning
├── requirements.txt
├── ARCHITECTURE.md            # this file
│
├── graph/
│   ├── __init__.py            # re-exports AgentState + compiled graph
│   ├── state.py                # AgentState TypedDict — the shared state schema
│   └── financial_agent_graph.py  # graph wiring: nodes, edges, conditional routing
│
├── nodes/                      # one file per graph node
│   ├── __init__.py             # exports every node function
│   ├── query_elaboration.py    # Node 1: enrich the raw query, extract ticker
│   ├── roadmap_planner.py      # Node 2: LLM plans tool sequence + report outline
│   ├── fetch_executor.py       # Node 3: deterministic fetch-tool execution
│   ├── compute_executor.py     # Node 4: forced-tool-choice calculator execution
│   ├── replanner.py            # shared recovery node (reached from 3 or 4)
│   ├── context_router.py       # Node 5: routes gathered context to report sections
│   ├── report_generator.py     # Node 6: writes the report section by section
│   └── report_saver.py         # Node 7: saves .md + .docx to disk
│
├── tools/                      # the actual data/compute functions
│   ├── __init__.py             # ALL_TOOLS, TOOLS_BY_NAME, FETCH_TOOL_FUNCTIONS,
│   │                           #   TOOL_DESCRIPTIONS — the tool registry
│   ├── sec_filing.py           # SEC EDGAR (edgartools + XBRL companyfacts API)
│   ├── financial_data.py       # yfinance + Alpha Vantage financials
│   ├── earnings_transcript.py  # Alpha Vantage earnings call transcripts
│   ├── web_search.py           # Tavily web search
│   └── calculator.py           # financial metric calculator (flat schema)
│
├── utils/
│   ├── __init__.py
│   ├── groq_client.py          # ChatGroq factory, model fallback, retry, tool_choice
│   ├── error_classifier.py     # shared 4-category failure classifier
│   └── helpers.py              # small text/JSON utilities
│
├── Ui/
│   └── app.py                  # Streamlit dashboard (streams the same graph)
│
├── sec_data/                   # edgartools' local filing cache (generated)
└── output/                     # generated reports land here (.md + .docx)
```

## Pipeline overview

```
query_elaboration → roadmap_planner → fetch_executor → compute_executor
    → context_router → report_generator → report_saver → END
```

Either executor can detour through a shared `replanner` node on failure:

```
fetch_executor ──needs_replan──> replanner ──phase="fetch"───> fetch_executor
     │ no                             │ phase="compute"
     ▼                                ▼
compute_executor <──────────────(back into loop)
     │ no
     ▼
context_router → report_generator → report_saver → END
```

### 1. `query_elaboration`
Extracts the ticker and expands the raw question into a fuller research
brief. Unchanged from the original design.

### 2. `roadmap_planner`
One LLM call produces two things as JSON:
- `research_plan`: an ordered list of tool-call steps (tool name, args, purpose)
- `report_outline`: the report's sections, each declaring a `data_sources`
  list — which tools that section's content should come from. This
  metadata is what `context_router` uses later instead of keyword scoring.

No fixed page-count target — depth follows what the data actually supports.

### 3. `fetch_executor` — deterministic, no LLM per step
Fetch tools (`sec_filing_search`, `web_search`, `earnings_transcript`,
`financial_data_api`) have arguments that are fully known at plan time
(ticker, filing type, year, quarter...). There's nothing for a model to
decide, so this node just calls the real Python functions directly off the
plan. This removes the old plan/execution drift bug entirely for these
steps — there's no free-form decision left to drift from.

Each result is run through `utils.error_classifier.classify_result()`:

| Category | Meaning | Handling |
|---|---|---|
| `OK` | Genuine success | Accept, tag `success` |
| `LEGITIMATE_EMPTY` | Valid "nothing here" answer (e.g. no filings for this ticker) | Accept, tag as unavailable, never retried |
| `BAD_ARGS` | Tool is fine, call was malformed | Retry once locally (`config.LOCAL_RETRY_LIMIT`) |
| `TRANSIENT` | One-off timeout/network hiccup | Retry once locally |
| `SYSTEMIC` | Tool is broken for the rest of the run (missing dependency, bad/missing API key, persistent rate limit) | Blacklist the tool (`disabled_tools`), stop this pass, escalate to `replanner` — but only if the global replan budget isn't already spent |

### 4. `compute_executor` — forced tool_choice, LLM does real work here
`calculator`'s inputs (e.g. a company's actual revenue figures) don't exist
at plan time — they only exist after `fetch_executor` has run. So this step
genuinely needs an LLM to read the accumulated context and fill in real
numbers. To remove the *other* kind of drift risk, `tool_choice` is forced
to the exact planned tool name via `get_llm(..., tool_choice=tool_name)` —
the model can only fill in arguments, it can't call something else or skip
the call. Same shared classifier and retry/escalate logic as the fetch
executor.

`config.COMPUTE_TOOLS` is the single switch that decides which tools go
through this path vs. the deterministic one — currently just `calculator`,
but any future derived-metric tool added there gets this same treatment for
free.

### `replanner` — shared recovery node
Only reached for `SYSTEMIC` failures (bad-args and transient failures never
get here — they're cheaper to fix with a local retry and don't need a
strategy change). Given the original goal, the disabled-tools list, which
steps already succeeded, and the specific failure, it produces a **revised
list of remaining steps** — not a single-step patch — because a systemic
failure invalidates every future step that still references the now-dead
tool, not just the one that failed. It either reroutes those steps to a
different tool or drops them with an explicit note.

Bounded by `config.GLOBAL_REPLAN_BUDGET` — a budget shared across both
executors for the whole run, not per-step. Once exhausted, further systemic
failures are just tagged and the run continues instead of looping.

If the replanner's own LLM call fails to produce valid JSON, it falls back
to a safe deterministic action: drop any remaining step touching a disabled
tool and note the gap, rather than risk repeating the same failure.

### 5. `context_router`
Replaces the old TF-IDF keyword-scoring `rag_retriever`. That approach
scored relevance as `matching keywords / total tokens in chunk` — which
systematically penalizes dense numeric/tabular data (a table of XBRL
figures has almost no natural-language overlap with a query like "earnings
revenue EPS"), so financial numbers routinely lost to prose passages.

Since every context chunk is already tagged with which tool produced it,
and every section already declares its `data_sources`, this node routes
deterministically: pull each section's declared tools' successful output
directly, no scoring involved. Keyword scoring (same technique as before)
is kept only as a fallback for sections without `data_sources` metadata, or
chunks from tools no section explicitly claimed.

If a section depends on a tool that failed or was disabled, a
`[Data unavailable] ...` line is inserted directly into that section's own
context — not a generic global disclaimer — so the report can name the
specific gap in the specific place it matters.

### 6. `report_generator`
Writes each section via LLM, instructed to state data gaps plainly (using
the `[Data unavailable]` markers) rather than silently omit them or
hallucinate around them. No fixed page-count target.

### 7. `report_saver`
Saves `.md` and `.docx` to `output/`. Unchanged.

## Shared building blocks

- **`config.py`** — every retry/replan/cap number lives here once:
  `LOCAL_RETRY_LIMIT`, `GLOBAL_REPLAN_BUDGET`, `COMPUTE_STEP_CAP`,
  `COMPUTE_TOOLS`, `GRAPH_RECURSION_LIMIT`.
- **`utils/error_classifier.py`** — the four-category classifier used
  identically by both executors, so there's one place that knows what a
  failure means, not one ad-hoc check per node.
- **`tools/__init__.py`** — `FETCH_TOOL_FUNCTIONS` (plain callables, used by
  the deterministic fetch executor) vs. `TOOLS_BY_NAME` / `ALL_TOOLS`
  (LangChain-wrapped, used for LLM binding in the compute executor). The
  calculator tool exposes every metric's real named parameters instead of
  `**kwargs` — LangChain's `@tool` decorator can't turn `**kwargs` into
  individual JSON schema fields, which was the root cause of every
  calculator call failing before this fix.

## State schema (`graph/state.py`)

- `context_chunks: list[dict]` — each entry: `{step, tool, status, content,
  retries_used}`. Previously a flat `list[str]`, which couldn't distinguish
  a real result from an error message once it reached the report.
- `disabled_tools: list[str]` — tools blacklisted this run.
- `replan_count: int` — checked against `config.GLOBAL_REPLAN_BUDGET`.
- `needs_replan`, `replan_reason`, `replan_phase` — transient routing
  signals an executor sets to detour through the replanner, and the
  replanner reads (`replan_phase`) to know which executor to hand control
  back to.

## What changed vs. the original design, and why

| Old symptom | Root cause | Fix |
|---|---|---|
| Calculator always crashed | `**kwargs` collapses to one opaque schema field; LLM nested args inside it | Flat named parameters for every metric |
| SEC data missing | `edgartools` not installed in the runtime env; XBRL only fetched for 10-K | Installed dependency; extended XBRL fetch to 10-Q |
| Plan said one thing, execution log showed another | LLM was asked to re-decide something the planner already decided, and completion was marked by the *planned* tool name regardless of what actually ran | Fetch tools execute deterministically off the plan; compute tools use forced `tool_choice` |
| Failed steps silently marked "done" | No result validation — any tool response was accepted | Four-category error classifier + local retry + replan escalation |
| Financial numbers missing from sections despite being fetched | TF-IDF keyword scoring penalizes numeric/tabular text | Metadata-based routing off each section's declared `data_sources` |
| Report was a fixed 15–16 pages regardless of available data | Hardcoded in both `roadmap_planner` and `report_generator` prompts | Removed from both; depth now follows data |
| Missing API key crashed the whole run before it started | `main.py` hard-exited on any missing key | Only `GROQ_API_KEY` is a hard requirement; others soft-warn and pre-seed `disabled_tools` |
| Two full copies of every tool + graph file, one silently dead | Confusion between `nodes/` and `tools/` package trees | Dead duplicates deleted; `nodes/__init__.py` only exports node functions now |

## Known limitations / good next steps

- No live end-to-end test has been run yet with real API keys — all fixes
  were verified in isolation (unit-level) since this dev environment has no
  network access to Groq, SEC EDGAR, Tavily, or Alpha Vantage.
- `calculator`'s required-args list (`tools/calculator.py:METRIC_REQUIRED_ARGS`)
  is manually maintained — adding a new metric means updating both the
  calculation function and this dict.
- `context_router`'s keyword-scoring fallback still uses the older
  TF-IDF-style technique for sections without `data_sources` metadata — fine
  as a fallback, but worth prompting `roadmap_planner` harder to always
  populate `data_sources` if this path gets hit often in practice.
- `sec_data/` (edgartools' local filing cache) and `output/` are runtime
  artifacts, not source — safe to `.gitignore`.
