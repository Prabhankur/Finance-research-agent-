# ── Dockerfile — Autonomous Financial Research Agent ─────────────────────────
#
# Build:  docker build -t financial-agent .
# Run:    docker run -p 8000:8000 --env-file .env financial-agent
# Dev:    docker-compose up
#
# WHY python:3.11-slim and not a full image:
#   - slim is ~150MB vs ~900MB for the full image
#   - all our dependencies install cleanly on Debian slim
#   - no CUDA needed — we call a hosted LLM (Groq) via API, not run one locally
#
# Stage 1: dependency builder (separate stage so Docker caches the pip install
# layer independently from the application code — changing app code does not
# trigger a full re-install of all packages)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed to compile some Python packages
# (psycopg2-binary is pre-compiled so libpq-dev is not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker layer cache: if requirements.txt has not
# changed, this whole layer is reused even if application code changed
COPY requirements.txt .

# Install into a specific prefix so we can copy just the installed packages
# into the final stage cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Create the directories the app writes to at runtime.
# output/  → SQLite database (financial_knowledge.db) + generated reports (.docx/.md)
# sec_data/→ SEC EDGAR downloads (cached filing text)
# These are created here so the container does not run as root to create them,
# and so Docker volumes can be mounted over them for persistence.
RUN mkdir -p output sec_data && chmod 777 output sec_data

# Copy application code
# .dockerignore excludes: output/*.db, .env, __pycache__, *.pyc, tests/
COPY . .

# ── Environment variables ────────────────────────────────────────────────────
# ALL secrets are injected at runtime via --env-file .env or docker-compose.
# NOTHING secret is baked into this image.
#
# Required (agent will not start without these):
#   GROQ_API_KEY          — Groq LLM API key
#
# Optional (agent degrades gracefully if missing):
#   ALPHA_VANTAGE_API_KEY — for earnings transcripts and AV financials
#   TAVILY_API_KEY        — for web search tool
#   LANGCHAIN_API_KEY     — for LangSmith tracing
#   LANGCHAIN_PROJECT     — LangSmith project name (default: financial-agent)
#
# Database (default SQLite, no config needed):
#   DATABASE_URL          — set to postgresql://... to switch to Postgres
#
# Feature flags:
#   EVAL_ENABLED          — true/false, default true (set false to save tokens)
#   GROQ_MODEL            — pin a specific model, default uses live catalog
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANGCHAIN_TRACING_V2=true \
    LANGCHAIN_PROJECT=financial-agent

# Expose the FastAPI port
EXPOSE 8000

# Health check — hits the /api/health endpoint every 30s.
# FastAPI returns 200 with {"status":"ok"} or {"status":"degraded"} if
# the GROQ key is missing. The health check considers both as healthy
# (the app is running) so Docker does not restart a misconfigured container
# in a loop — the operator can see the degraded status and fix the key.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" \
    || exit 1

# Non-root user for security — never run application containers as root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Start the FastAPI server
# --host 0.0.0.0 is required inside Docker (default 127.0.0.1 is unreachable from outside the container)
# --workers 1 because: SQLite is single-writer, LangGraph checkpointer is
#   in-process, and our ThreadPoolExecutor handles concurrency within one
#   worker. Multiple workers would corrupt the SQLite DB.
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
