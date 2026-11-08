"""
database/cache_service.py  –  TTL cache + provenance, the layer tools call.

This is the piece that actually reduces API calls. The pattern every tool
follows:

    check DB
       ├── fresh?  -> return it, no API call
       └── stale/missing -> call API -> normalise -> store -> return

Crucially the tool doesn't know or care which branch it took. That's the
point of putting this behind a function rather than sprinkling cache checks
through each tool: the caching policy lives in one place and can be changed
(or disabled entirely) without touching a single tool.

FAILURE POSTURE
Every function here swallows its own exceptions and degrades to "no cache".
A caching layer that can take down the agent is worse than no caching layer,
so a corrupt DB file, a locked table, or a missing Postgres server all
produce the same outcome: a cache miss, and the live API gets called exactly
as it did before this module existed.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from database.connection import execute, init_db, query_one
from database.schema import DEFAULT_TTL_SECONDS, TTL_SECONDS

init_db()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def canonical_args(args: dict) -> dict:
    """
    The arguments as they should be COMPARED, not as they were typed.

    A hash key is exact, so without this "msft", "MSFT" and " Msft " were three
    different cache entries for the same request (and a lower-case spelling also
    created a second company row downstream). A model will happily produce any of
    them. Only the arguments where case and padding cannot change the meaning are
    touched: ticker (upper-cased, trimmed) and statement_type / period
    (lower-cased, trimmed). Everything else is left exactly as given — a filing
    type such as "10-K", for instance, is not ours to reinterpret.
    """
    out = dict(args)
    for name, value in args.items():
        if not isinstance(value, str):
            continue
        if name == "ticker":
            out[name] = value.strip().upper()
        elif name in ("statement_type", "period"):
            out[name] = value.strip().lower()
    return out


def make_cache_key(tool_name: str, args: dict) -> str:
    """
    Stable key for a (tool, arguments) pair.

    Args are sorted before hashing so {"ticker":"MSFT","years":2} and
    {"years":2,"ticker":"MSFT"} — the same call, and something an LLM will
    absolutely produce both ways — hit the same cache entry rather than
    counting as two distinct calls. They are also canonicalized first (see
    canonical_args), so spelling variants of the same request share one entry.
    """
    normalised = json.dumps(canonical_args(args), sort_keys=True, default=str)
    digest = hashlib.sha256(f"{tool_name}|{normalised}".encode()).hexdigest()[:32]
    return f"{tool_name}:{digest}"


def get_cached(cache_key: str) -> Optional[str]:
    """Returns the cached payload if present AND unexpired, else None."""
    try:
        row = query_one(
            "SELECT payload, expires_at FROM response_cache WHERE cache_key = ?",
            (cache_key,),
        )
        if not row:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at <= _now():
            return None
        return row["payload"]
    except Exception as exc:
        print(f"[Cache] Read failed ({exc}) — treating as a miss.")
        return None


def set_cached(cache_key: str, tool_name: str, payload: str, ttl_class: str) -> None:
    """Stores a payload under a TTL class (see schema.TTL_SECONDS)."""
    ttl = TTL_SECONDS.get(ttl_class, DEFAULT_TTL_SECONDS)
    now = _now()
    try:
        execute(
            """
            INSERT INTO response_cache (cache_key, tool_name, payload, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                payload    = excluded.payload,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
            """,
            (cache_key, tool_name, payload, _iso(now), _iso(now + timedelta(seconds=ttl))),
        )
    except Exception as exc:
        print(f"[Cache] Write failed ({exc}) — continuing without caching.")
    _maybe_purge()


# Expired rows are ignored on read, so purging is not a correctness matter — but
# without it response_cache only ever grew. Cache WRITES trigger it, at most once
# per interval: a write is the moment the table is growing, it needs no scheduler
# or background thread, and one indexed DELETE per hour is negligible.
_PURGE_INTERVAL_S = 3600
_last_purge = 0.0
_purge_lock = threading.Lock()


def _maybe_purge() -> None:
    """Runs purge_expired at most once per _PURGE_INTERVAL_S. Never raises."""
    global _last_purge
    try:
        now = time.time()
        if now - _last_purge < _PURGE_INTERVAL_S:
            return
        with _purge_lock:
            if now - _last_purge < _PURGE_INTERVAL_S:    # another thread just did it
                return
            _last_purge = now
        removed = purge_expired()
        if removed:
            print(f"[Cache] Purged expired cache entries ({removed}).")
    except Exception as exc:
        print(f"[Cache] Purge skipped ({exc}).")


def cached_call(
    tool_name: str,
    args: dict,
    ttl_class: str,
    fetch: Callable[[], str],
    normalize: Optional[Callable[[str, int | None], None]] = None,
    provider: Optional[str] = None,
    source_type: Optional[str] = None,
) -> tuple[str, bool]:
    """
    The main entry point. Returns (payload, was_cached).

    The `was_cached` flag is returned rather than hidden so callers can
    surface it — knowing an answer came from a 6-day-old cache rather than a
    live call is exactly the kind of thing a financial analyst needs to see,
    and it's also how you'd verify the cache is actually working instead of
    silently missing on every call.

    `normalize` and `provider` are OPTIONAL — they turn a plain caching call
    into "cache + provenance + typed row extraction". When both are supplied,
    a live fetch also:
      - creates a sources row (provider, retrieved_at) → source_id
      - calls normalize(payload, source_id) so the tool's response gets
        parsed into typed rows in the knowledge-layer tables
    See database/normalizer.py for the parsers. This deliberately runs AFTER
    the payload has been returned to the caller — normalization failing must
    not affect the tool's return value; the row just doesn't get written.
    """
    key = make_cache_key(tool_name, args)

    hit = _probe(key, tool_name, ttl_class)
    if hit is not None:
        return hit, True
    print(f"[Cache] MISS {tool_name} ({ttl_class}) — calling live API")

    # REQUEST COALESCING (single-flight). Without this, two identical requests
    # that arrive together — e.g. the model asking for the same company twice in
    # one parallel round, or two browser tabs — both miss before either has
    # stored its result, and BOTH call the live API (a "cache stampede"). On a
    # provider capped at 25 calls a day that is a real cost. The first thread
    # for a key becomes the LEADER and does the fetch; identical requests that
    # arrive meanwhile wait for its result instead of fetching again.
    while True:
        flight, leader = _claim(key)
        if leader:
            try:
                # Double-checked: another leader may have finished between our
                # first probe and winning leadership.
                hit = _probe(key, tool_name, ttl_class, announce=False)
                if hit is not None:
                    flight.payload = hit
                    return hit, True
                payload = _fetch_and_store(
                    key, tool_name, ttl_class, fetch, normalize, provider, source_type
                )
                flight.payload = payload      # shared with waiters, even an error text
                return payload, False
            finally:
                _release(key, flight)         # always — a failed fetch must not strand waiters

        print(f"[Cache] COALESCED {tool_name} — an identical request is already in flight; waiting")
        finished = flight.done.wait(timeout=_COALESCE_WAIT_S)
        if finished and flight.payload is not None:
            return flight.payload, True       # answered by the leader's call: no API call of our own
        if not finished:
            # The leader is taking unusually long. Stop waiting rather than hang
            # this request behind it; fetch independently.
            print(f"[Cache] Coalesced wait for {tool_name} timed out — fetching independently.")
            return _fetch_and_store(
                key, tool_name, ttl_class, fetch, normalize, provider, source_type
            ), False
        # The leader raised, so there is no result to share. Loop: become the
        # leader (or follow whoever beat us to it) and try the fetch ourselves.


# ── coalescing machinery ──────────────────────────────────────────────────────
# Process-local on purpose: it coordinates THREADS (the chat tool node's
# ThreadPoolExecutor, the report fetch groups, concurrent API requests under a
# single uvicorn worker). Several worker PROCESSES would each have their own
# table; coordinating across them needs a shared lock (a database advisory lock
# or Redis) — the natural next step if the app is scaled out.
_COALESCE_WAIT_S = 90.0
_INFLIGHT: dict[str, "_Flight"] = {}
_INFLIGHT_LOCK = threading.Lock()


class _Flight:
    """One in-progress fetch that identical requests can wait on."""
    __slots__ = ("done", "payload")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.payload: Optional[str] = None


def _claim(key: str) -> tuple["_Flight", bool]:
    """(flight, is_leader). The first caller for a key leads; the rest follow."""
    with _INFLIGHT_LOCK:
        flight = _INFLIGHT.get(key)
        if flight is None:
            flight = _INFLIGHT[key] = _Flight()
            return flight, True
        return flight, False


def _release(key: str, flight: "_Flight") -> None:
    with _INFLIGHT_LOCK:
        if _INFLIGHT.get(key) is flight:
            del _INFLIGHT[key]
    flight.done.set()


def _probe(key: str, tool_name: str, ttl_class: str, announce: bool = True) -> Optional[str]:
    """Cache lookup that can never raise. A cache that breaks the agent is worse
    than no cache, so any failure here just means "go to the live API"."""
    try:
        hit = get_cached(key)
        if hit is not None and announce:
            print(f"[Cache] HIT {tool_name} ({ttl_class})")
        return hit
    except Exception as exc:
        print(f"[Cache] Lookup path failed ({exc}) — going straight to the live API.")
        return None


def _fetch_and_store(key, tool_name, ttl_class, fetch, normalize, provider, source_type) -> str:
    """The live fetch plus the store / provenance / normalize steps."""
    payload = fetch()

    # Never cache an error response. Caching a transient failure for 7 days
    # would turn a blip into a week-long outage for that query.
    try:
        if payload and not _looks_like_error(payload):
            set_cached(key, tool_name, payload, ttl_class)

            # Structured normalization runs strictly AFTER caching, and its own
            # failures are swallowed (per normalize's contract). The tool caller
            # sees the same string either way — normalization is purely additive.
            # Provenance only needs `provider`; `normalize` is a separate,
            # optional upgrade (news / transcripts have no typed rows to
            # extract but still deserve "which provider, retrieved when").
            if provider is not None:
                try:
                    source_id = record_source(provider=provider, source_type=source_type or "api")
                    if normalize is not None:
                        normalize(payload, source_id)
                except Exception as exc:
                    print(f"[Cache] Provenance/normalize step failed ({exc}) — cache still populated.")
    except Exception as exc:
        print(f"[Cache] Store path failed ({exc}) — result still returned.")

    return payload


def _looks_like_error(payload: str) -> bool:
    head = payload[:400].lower()
    return any(
        marker in head
        for marker in ("error:", "no data", "not found", "rate limit", "unauthorized", "failed")
    )


def record_source(
    provider: str,
    source_type: str,
    url: str | None = None,
    published_at: str | None = None,
    content_hash: str | None = None,
) -> Optional[int]:
    """
    Records where a fact came from and returns its source id.

    Called on every successful fetch so that stored facts can point back at
    a specific retrieval event. This is the row that makes it possible to
    later say "SEC 10-K, retrieved 2026-08-31" next to a number instead of
    just asserting the number.
    """
    try:
        return execute(
            """
            INSERT INTO sources (source_type, provider, url, retrieved_at, published_at, content_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source_type, provider, url, _iso(_now()), published_at, content_hash),
        )
    except Exception as exc:
        print(f"[DB] Could not record source ({exc}).")
        return None


def cache_stats() -> dict[str, Any]:
    """Counts for the /api/health endpoint — proves the cache is being used."""
    try:
        total = query_one("SELECT COUNT(*) AS n FROM response_cache") or {"n": 0}
        live = query_one(
            "SELECT COUNT(*) AS n FROM response_cache WHERE expires_at > ?", (_iso(_now()),)
        ) or {"n": 0}
        sources = query_one("SELECT COUNT(*) AS n FROM sources") or {"n": 0}
        return {
            "cached_entries": total["n"],
            "unexpired_entries": live["n"],
            "sources_recorded": sources["n"],
        }
    except Exception as exc:
        return {"error": str(exc)}


def purge_expired() -> int:
    """Deletes expired rows. Called automatically from cache writes, at most once
    an hour (see _maybe_purge); expired entries are ignored on read either way, so
    this keeps the table from growing without bound rather than guarding correctness."""
    try:
        return execute("DELETE FROM response_cache WHERE expires_at <= ?", (_iso(_now()),))
    except Exception:
        return 0
