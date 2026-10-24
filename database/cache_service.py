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
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from database.connection import execute, init_db, query_one
from database.schema import DEFAULT_TTL_SECONDS, TTL_SECONDS

init_db()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def make_cache_key(tool_name: str, args: dict) -> str:
    """
    Stable key for a (tool, arguments) pair.

    Args are sorted before hashing so {"ticker":"MSFT","years":2} and
    {"years":2,"ticker":"MSFT"} — the same call, and something an LLM will
    absolutely produce both ways — hit the same cache entry rather than
    counting as two distinct calls.
    """
    normalised = json.dumps(args, sort_keys=True, default=str)
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

    # The whole cache path is wrapped, not just the individual read/write
    # helpers. A test caught this gap: guarding get_cached/set_cached
    # internally still left make_cache_key and the branching logic here
    # unguarded, so a failure in those propagated out and would have taken
    # down a tool call that could have simply gone to the live API. A
    # caching layer that can break the agent is worse than no caching layer.
    try:
        hit = get_cached(key)
        if hit is not None:
            print(f"[Cache] HIT {tool_name} ({ttl_class})")
            return hit, True
        print(f"[Cache] MISS {tool_name} ({ttl_class}) — calling live API")
    except Exception as exc:
        print(f"[Cache] Lookup path failed ({exc}) — going straight to the live API.")

    payload = fetch()

    # Never cache an error response. Caching a transient failure for 7 days
    # would turn a blip into a week-long outage for that query.
    try:
        if payload and not _looks_like_error(payload):
            set_cached(key, tool_name, payload, ttl_class)

            # Structured normalization runs strictly AFTER caching, and its
            # own failures are swallowed (per normalize's contract). The
            # tool caller sees the same string either way — normalization
            # is purely additive: it upgrades response_cache's "seen this
            # call recently" into typed knowledge-layer rows the DB can
            # answer real queries against.
            # Provenance recording only needs `provider` — `normalize` is a
            # SEPARATE, optional upgrade on top of it. This distinction
            # matters for unstructured content (news, transcripts): there's
            # no meaningful typed row to extract from a news snippet or a
            # call transcript, but "which provider, retrieved when" is still
            # real, useful provenance. Requiring both used to mean
            # unstructured tools got NO provenance at all, silently.
            if provider is not None:
                try:
                    source_id = record_source(provider=provider, source_type=source_type or "api")
                    if normalize is not None:
                        normalize(payload, source_id)
                except Exception as exc:
                    print(f"[Cache] Provenance/normalize step failed ({exc}) — cache still populated.")
    except Exception as exc:
        print(f"[Cache] Store path failed ({exc}) — result still returned.")

    return payload, False


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
    """Deletes expired rows. Nothing calls this automatically yet — expired
    entries are simply ignored on read, so this is a maintenance helper
    rather than a correctness requirement."""
    try:
        return execute("DELETE FROM response_cache WHERE expires_at <= ?", (_iso(_now()),))
    except Exception:
        return 0
