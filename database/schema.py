"""
database/schema.py  –  Tier 1 financial knowledge layer.

Portable SQL that runs on BOTH SQLite (zero-setup local dev) and PostgreSQL
(the eventual target), selected purely by the DATABASE_URL env var. No ORM:
the query surface here is a couple of dozen well-understood statements, and
SQLAlchemy would add a dependency, a migration story, and a layer of
indirection for no gain at this size.

WHAT THIS IS FOR
Two problems, one schema:

  1. Caching. Every question currently re-hits yfinance / Alpha Vantage /
     FRED even when the answer is a figure that cannot have changed — a
     company's FY2024 revenue is fixed forever once filed. With Alpha
     Vantage capped at 25 calls/day, re-fetching immutable data is the main
     thing burning that budget.

  2. Provenance. Every stored fact points at a `sources` row recording where
     it came from and when it was retrieved. That's the foundation for the
     evidence-grading and contradiction-detection work: you cannot flag
     "SEC says $391.0B but yfinance says $391.04B" until both numbers are
     stored as separate, individually-attributed facts rather than
     collapsed into one text blob.

WHY NOT STORE RAW API RESPONSES
Storing provider payloads verbatim just relocates the existing problem —
you'd still be regex-scraping numbers out of text, only now out of a
database instead of an API response. Normalising to typed columns at write
time is what makes `SELECT revenue FROM income_statements WHERE ...`
possible, and that query is the entire point.

PERIOD SEMANTICS
`period_end` (a real date) is the source of truth for which period a row
describes; `fiscal_period` ("FY2024", "Q2-2025") is a human label derived
from it. Storing both is deliberate: fiscal years don't align to calendar
years (Apple's FY2024 ended 2024-09-28, Microsoft's FY2026 ended
2026-06-30), and comparing companies on a label alone silently compares
different windows of time.
"""

from __future__ import annotations

# Tables are created in dependency order; every statement is IF NOT EXISTS so
# init is idempotent and safe to call on every process start.
SCHEMA_STATEMENTS: list[str] = [
    # ── Provenance ───────────────────────────────────────────────────────────
    # Every fact table carries source_id -> sources.id. Without this, two
    # providers disagreeing is undetectable: one silently overwrites the
    # other and nothing records that a conflict ever existed.
    """
    CREATE TABLE IF NOT EXISTS sources (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        source_type   TEXT NOT NULL,          -- 'api' | 'filing' | 'web' | 'macro'
        provider      TEXT NOT NULL,          -- 'yfinance' | 'alpha_vantage' | 'sec' | 'fred' | 'tavily'
        url           TEXT,
        retrieved_at  TEXT NOT NULL,
        published_at  TEXT,
        content_hash  TEXT,
        data_version  TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sources_provider ON sources(provider, retrieved_at DESC)",

    # ── Company master ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS companies (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT NOT NULL UNIQUE,
        name        TEXT,
        exchange    TEXT,
        country     TEXT,
        sector      TEXT,
        industry    TEXT,
        -- Trading currency (what the share is quoted in).
        currency    TEXT,
        -- Reporting currency (what the financial statements are in). A
        -- different fact: an ADR can trade in USD yet report in INR.
        financial_currency TEXT,
        cik         TEXT,
        employees   INTEGER,
        website     TEXT,
        city        TEXT,
        state       TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker)",

    # ── Financial statements ─────────────────────────────────────────────────
    # UNIQUE(company_id, period_end, period_type, source_id) rather than
    # UNIQUE(company_id, period_end): the same period from DIFFERENT sources
    # must be able to coexist as separate rows. That's precisely what makes
    # reconciliation possible later — collapsing them would destroy the
    # disagreement before anyone could see it.
    """
    CREATE TABLE IF NOT EXISTS income_statements (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id        INTEGER NOT NULL REFERENCES companies(id),
        fiscal_period     TEXT,
        period_end        TEXT NOT NULL,
        period_type       TEXT NOT NULL,      -- 'annual' | 'quarterly'
        revenue           REAL,
        cost_of_revenue   REAL,
        gross_profit      REAL,
        operating_income  REAL,
        net_income        REAL,
        eps_basic         REAL,
        eps_diluted       REAL,
        ebitda            REAL,
        rd_expense        REAL,
        currency          TEXT,
        source_id         INTEGER REFERENCES sources(id),
        updated_at        TEXT NOT NULL,
        UNIQUE(company_id, period_end, period_type, source_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_income_lookup ON income_statements(company_id, period_type, period_end DESC)",

    """
    CREATE TABLE IF NOT EXISTS balance_sheets (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id           INTEGER NOT NULL REFERENCES companies(id),
        fiscal_period        TEXT,
        period_end           TEXT NOT NULL,
        period_type          TEXT NOT NULL,
        cash                 REAL,
        total_assets         REAL,
        total_liabilities    REAL,
        total_equity         REAL,
        total_debt           REAL,
        -- SEC's LongTermDebt concept. NOT the same quantity as total_debt
        -- (yfinance 'Total Debt' also includes short-term debt and leases);
        -- storing it in total_debt made the contradiction detector compare
        -- two different measures and report false conflicts.
        long_term_debt       REAL,
        current_assets       REAL,
        current_liabilities  REAL,
        retained_earnings    REAL,
        currency             TEXT,
        source_id            INTEGER REFERENCES sources(id),
        updated_at           TEXT NOT NULL,
        UNIQUE(company_id, period_end, period_type, source_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_balance_lookup ON balance_sheets(company_id, period_type, period_end DESC)",

    """
    CREATE TABLE IF NOT EXISTS cash_flow_statements (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id           INTEGER NOT NULL REFERENCES companies(id),
        fiscal_period        TEXT,
        period_end           TEXT NOT NULL,
        period_type          TEXT NOT NULL,
        operating_cash_flow  REAL,
        capital_expenditure  REAL,
        investing_cash_flow  REAL,
        financing_cash_flow  REAL,
        free_cash_flow       REAL,
        -- Printed by tools/financial_data.py ("Dividends Paid", "Share
        -- Buybacks") but previously had NO column, so the normalizer threw
        -- them away. Stored as the provider reports them: negative =
        -- cash OUT, same convention as capital_expenditure.
        dividends_paid       REAL,
        share_repurchases    REAL,
        currency             TEXT,
        source_id            INTEGER REFERENCES sources(id),
        updated_at           TEXT NOT NULL,
        UNIQUE(company_id, period_end, period_type, source_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cashflow_lookup ON cash_flow_statements(company_id, period_type, period_end DESC)",

    # ── Market data ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS market_prices (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id      INTEGER NOT NULL REFERENCES companies(id),
        date            TEXT NOT NULL,
        open            REAL,
        high            REAL,
        low             REAL,
        close           REAL,
        adjusted_close  REAL,
        volume          REAL,
        source_id       INTEGER REFERENCES sources(id),
        UNIQUE(company_id, date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_prices_lookup ON market_prices(company_id, date DESC)",

    # Valuation multiples are POINT-IN-TIME, keyed by date, unlike statements
    # which are keyed by period. A P/E from last week is a different fact
    # from a P/E today — not a correction of it — so these accumulate as
    # snapshots rather than being overwritten.
    """
    CREATE TABLE IF NOT EXISTS valuation_snapshots (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        company_id        INTEGER NOT NULL REFERENCES companies(id),
        date              TEXT NOT NULL,
        market_cap        REAL,
        enterprise_value  REAL,
        pe                REAL,
        forward_pe        REAL,
        price_to_sales    REAL,
        price_to_book     REAL,
        ev_ebitda         REAL,
        peg               REAL,
        dividend_yield    REAL,
        beta              REAL,
        -- Remaining fields of yfinance's market-overview response. They were
        -- returned to the model on every overview call but had nowhere to
        -- live in the typed layer.
        price             REAL,
        week52_high       REAL,
        week52_low        REAL,
        ma_50             REAL,
        ma_200            REAL,
        profit_margin     REAL,
        roe               REAL,
        debt_to_equity    REAL,
        revenue_ttm       REAL,
        net_income_ttm    REAL,
        source_id         INTEGER REFERENCES sources(id),
        UNIQUE(company_id, date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_valuation_lookup ON valuation_snapshots(company_id, date DESC)",

    # ── Macro ────────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS macro_series (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        series_code  TEXT NOT NULL UNIQUE,
        name         TEXT,
        country      TEXT,
        frequency    TEXT,
        unit         TEXT,
        -- 'Seasonally Adjusted Annual Rate' vs 'Not Seasonally Adjusted'.
        -- This distinction is exactly what a user hit when asking for GDP
        -- 'not as an annual rate' — the raw observations alone cannot tell
        -- two series apart without it.
        seasonal_adjustment TEXT,
        source       TEXT,
        updated_at   TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS macro_observations (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        series_id  INTEGER NOT NULL REFERENCES macro_series(id),
        date       TEXT NOT NULL,
        value      REAL,
        source_id  INTEGER REFERENCES sources(id),
        UNIQUE(series_id, date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_macro_obs_lookup ON macro_observations(series_id, date DESC)",

    # ── Raw response cache ───────────────────────────────────────────────────
    # Sits alongside the normalised tables rather than replacing them.
    # Normalised tables answer "what is MSFT's FY2026 revenue" with a typed
    # number; this answers "have we already made this exact API call
    # recently" for calls whose output isn't (yet) normalised into columns —
    # SEC filing text, news, transcripts. Together they cover every tool.
    """
    CREATE TABLE IF NOT EXISTS response_cache (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        cache_key   TEXT NOT NULL UNIQUE,
        tool_name   TEXT NOT NULL,
        payload     TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cache_key ON response_cache(cache_key)",
    "CREATE INDEX IF NOT EXISTS idx_cache_expiry ON response_cache(expires_at)",
]


# ── Migrations for databases created before a column existed ─────────────────
# `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists,
# so editing a CREATE statement alone never reaches an existing database file.
# Each entry is (table, column, sql_type); connection.init_db adds the column
# only if it is not already there (checked via the catalog, so it is safe to
# run on every start and portable across SQLite and Postgres).
SCHEMA_MIGRATIONS: list[tuple[str, str, str]] = [
    ("cash_flow_statements", "dividends_paid",      "REAL"),
    ("cash_flow_statements", "share_repurchases",   "REAL"),
    ("valuation_snapshots",  "price",               "REAL"),
    ("valuation_snapshots",  "week52_high",         "REAL"),
    ("valuation_snapshots",  "week52_low",          "REAL"),
    ("valuation_snapshots",  "ma_50",               "REAL"),
    ("valuation_snapshots",  "ma_200",              "REAL"),
    ("valuation_snapshots",  "profit_margin",       "REAL"),
    ("valuation_snapshots",  "roe",                 "REAL"),
    ("valuation_snapshots",  "debt_to_equity",      "REAL"),
    ("valuation_snapshots",  "revenue_ttm",         "REAL"),
    ("valuation_snapshots",  "net_income_ttm",      "REAL"),
    ("macro_series",         "seasonal_adjustment", "TEXT"),
    ("balance_sheets",       "long_term_debt",      "REAL"),
    ("companies",            "financial_currency",  "TEXT"),
]

# One-off corrections for rows written before a mapping was fixed. Each must
# be idempotent (a no-op once applied) because init_db runs on every start.
DATA_MIGRATIONS: list[str] = [
    # SEC rows used to put LongTermDebt into total_debt. Relocate it.
    """UPDATE balance_sheets
          SET long_term_debt = total_debt, total_debt = NULL
        WHERE total_debt IS NOT NULL AND long_term_debt IS NULL
          AND source_id IN (SELECT id FROM sources WHERE provider = 'sec')""",
    # income_statements: keep only the newest row per (company, period, type, provider).
    # Databases written before the supersede fix hold one extra copy of every
    # period per refetch. LEFT JOIN + COALESCE so rows with no source are kept.
    """DELETE FROM income_statements WHERE id NOT IN (
           SELECT MAX(t.id) FROM income_statements t LEFT JOIN sources s ON s.id = t.source_id
            GROUP BY t.company_id, t.period_end, t.period_type, COALESCE(s.provider, ''))""",
    # balance_sheets: keep only the newest row per (company, period, type, provider).
    # Databases written before the supersede fix hold one extra copy of every
    # period per refetch. LEFT JOIN + COALESCE so rows with no source are kept.
    """DELETE FROM balance_sheets WHERE id NOT IN (
           SELECT MAX(t.id) FROM balance_sheets t LEFT JOIN sources s ON s.id = t.source_id
            GROUP BY t.company_id, t.period_end, t.period_type, COALESCE(s.provider, ''))""",
    # cash_flow_statements: keep only the newest row per (company, period, type, provider).
    # Databases written before the supersede fix hold one extra copy of every
    # period per refetch. LEFT JOIN + COALESCE so rows with no source are kept.
    """DELETE FROM cash_flow_statements WHERE id NOT IN (
           SELECT MAX(t.id) FROM cash_flow_statements t LEFT JOIN sources s ON s.id = t.source_id
            GROUP BY t.company_id, t.period_end, t.period_type, COALESCE(s.provider, ''))""",
]


# ── Freshness policy ─────────────────────────────────────────────────────────
# TTL per data class, in seconds. The governing question for each is "how
# long until this value could legitimately change?", NOT "how stale can we
# tolerate" — which is why the spread is five orders of magnitude.
#
# Filed statements are immutable once filed (a 10-K's FY2024 revenue will
# never change), so their TTL is really "how long until the NEXT filing might
# supersede this as the latest period" — hence 7 days rather than infinity.
TTL_SECONDS: dict[str, int] = {
    "sec_filing":        30 * 24 * 3600,   # filings never change once filed
    "annual_financials":  7 * 24 * 3600,   # new annual data at most 4x/year
    "quarterly_financials": 24 * 3600,     # quarterly cadence, checked daily
    "company_profile":   14 * 24 * 3600,   # sector/employees move slowly
    "macro_series":           24 * 3600,   # FRED publishes daily at most
    "valuation":               3600,       # multiples move with the price
    "market_price":            3600,       # intraday staleness is acceptable
    "earnings_transcript": 30 * 24 * 3600, # transcript text is immutable
    "news":                    3600,       # "recent news" means recent
}

DEFAULT_TTL_SECONDS = 3600
