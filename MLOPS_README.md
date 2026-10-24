# MLOps Setup — Autonomous Financial Research Agent

## What is here and why

Three MLOps additions, each with a genuine reason to exist for this project:

---

## 1. Docker

**Files:** `Dockerfile`, `.dockerignore`, `docker-compose.yml`

**Why it makes sense for this project:**

The agent has real environment dependencies that are painful to set up manually:
- Python 3.11 with 15+ packages
- API keys for 4 external services
- A writable output directory for the SQLite database and generated reports
- Environment variable configuration for feature flags

Docker packages all of this into a single container that starts with one command.

**Run locally:**
```bash
# Copy the env template and fill in your API keys
cp .env.example .env

# SQLite mode (default — zero setup)
docker-compose up --build

# Postgres mode
docker-compose --profile postgres up --build

# Open the UI
open http://localhost:8000
```

**Design decisions worth explaining in an interview:**

- **Multi-stage build** — a `builder` stage installs packages, a `runtime` stage copies only what is needed. The final image is ~400MB instead of ~900MB.
- **Named volumes, not bind mounts** — `agent_output` and `agent_sec_data` are Docker-managed volumes. Bind mounts have permission issues on Windows and macOS with Docker Desktop.
- **`--workers 1`** in the CMD — SQLite cannot handle concurrent writers from multiple processes. One worker is correct. The `ThreadPoolExecutor` inside the app handles within-process concurrency for parallel tool calls.
- **Non-root user** — the container runs as `appuser` (uid 1000), not root. Standard security practice.
- **Health check** — Docker monitors `/api/health` every 30 seconds. The endpoint returns `{"status":"ok"}` or `{"status":"degraded"}` (if a key is missing). Both are considered healthy by the health check so Docker does not restart a misconfigured container in a loop.

---

## 2. GitHub Actions CI

**File:** `.github/workflows/ci.yml`

**Why it makes sense for this project:**

The test suite has already caught 3 real production bugs during development:
- The `psycopg2.lastrowid` bug that was invisible on SQLite and only appeared on live Postgres
- The calculator duplicate-schema bug where the wrapper was missing the `gross_profit` parameter
- The field-filter whitespace normalization bug (`"Total Revenue      :"` not matching `"Total Revenue:"`)

Automating the test run on every commit means these regressions cannot silently re-enter the codebase. If a future change breaks something the tests cover, the push fails before it reaches anyone.

**Three jobs:**

| Job | What it tests | Why |
|---|---|---|
| `test` | Full 176-test suite | Catches regressions in agent logic, caching, normalization, evaluator |
| `smoke-test` | DB init + FastAPI import | Catches schema errors and import failures on first startup |
| `docker-build` | Docker image build + health check | Catches Dockerfile errors and confirms the containerized app starts |

**No real API keys needed in CI** — the test suite uses `install_fake_tools()` to mock all external calls. Only a fake `GROQ_API_KEY` is needed to satisfy import checks. No actual money is spent running CI.

---

## 3. LangSmith (already integrated — framed as MLOps)

**Why this is the ML-specific observability layer:**

For a system built on LLM API calls rather than trained models, LangSmith is the equivalent of APM monitoring for a traditional backend. It provides:

- **Distributed tracing** — every agent run is a trace with child spans for each node (intent_router, chat_agent, tool calls, contradiction_detector, evaluator)
- **Token accounting** — input tokens, output tokens, and cost per run — the data behind the cost analysis in this project
- **Latency breakdown** — which node took longest? (Usually the live API fetches, not the LLM calls)
- **Failure inspection** — when rescue synthesis fires, LangSmith shows exactly which round triggered the tool_use_failed error and what the model's raw output was
- **Evaluation results** — LLM-as-judge scores are logged as feedback on each run, creating a queryable history of quality over time

This is MLOps observability for an LLM-powered system. The equivalent for a traditional ML pipeline would be MLflow experiment tracking — but MLflow is for teams training models and comparing runs. LangSmith is the correct tool for a system that calls hosted models via API.

---

## Interview framing

When asked "did you add any MLOps?":

> "Yes — three pieces. Docker containerizes the application with all its dependencies, including environment variable injection for API keys and named volumes for SQLite persistence. GitHub Actions runs my 176-test suite on every commit — that suite has caught real production bugs so I automated it rather than relying on remembering to run it manually. And LangSmith handles the ML-specific observability — every agent run is traced with token counts, tool-call latency, and failure points. For a system built on LLM API calls rather than trained models, that is what MLOps observability looks like."

---

## What was deliberately NOT added

| Tool | Why not |
|---|---|
| Kubernetes | Overkill — one container, one process, one user. The complexity would not be justified. |
| MLflow / DVC | For teams training models and tracking experiments. You are calling a hosted API, not training. |
| Prometheus / Grafana | A full metrics stack for a single SQLite-backed service is not defensible. LangSmith covers the ML-specific metrics that actually matter. |
| Model registry | You use hosted models (Groq). There is nothing to register — model selection happens via the live catalog endpoint. |
