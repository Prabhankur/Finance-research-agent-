"""
utils/groq_client.py  –  Central Groq LLM factory with model fallback.

All nodes import get_llm() from here instead of hardcoding the model name.
To change the model for the entire project, just update GROQ_MODEL in .env

Model selection order:
  1. GROQ_MODEL from .env, if set
  2. Live catalog from GET /v1/models (cached per process), filtered to
     known chat-capable models — this is ground truth for what the CURRENT
     API key can actually use
  3. Hardcoded _FALLBACK_MODELS list, if the catalog call fails
"""

from __future__ import annotations
import os
import re
from typing import Optional
from langchain_groq import ChatGroq

# Hardcoded fallback chain, used when the live catalog (see
# list_available_models) can't be reached. Ordered by preference:
# tool-calling reliability first, then size.
#
# A hardcoded list is inherently a guess about a catalog that changes on
# Groq's schedule, not ours — that's exactly how this project ended up
# calling a 404'd model. It's kept only as a last resort; the live catalog
# below is the real mechanism.
_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
]

# Models that exist in the catalog but are NOT chat/tool-calling models —
# filtered out of auto-selection so we never try to run a chat turn against
# a speech-to-text or moderation endpoint.
_NON_CHAT_MODEL_MARKERS = (
    "whisper", "tts", "guard", "embed", "distil-whisper", "playai",
)

# Preference order applied to whatever the live catalog returns. Anything
# matching an earlier prefix wins. Llama-70b-class first because this
# project's tool-calling prompts were built and tested against them;
# gpt-oss models are capable but have shown tool-calling quirks here
# (harmony-format token leakage into tool names, and emitting tool calls
# even when tools are disabled — see nodes/chat_agent.py's final-round
# handling).
_MODEL_PREFERENCE = (
    "llama-3.3-70b",
    "llama-3.1-70b",
    "llama3-70b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b",
    "llama3-8b",
    "gemma2",
)

# Process-level cache: the catalog is stable within a run, and this is on
# the hot path for every single get_llm() call.
_CATALOG_CACHE: Optional[list[str]] = None


def list_available_models(force_refresh: bool = False) -> list[str]:
    """
    Fetches the models this API key can actually use, via GET /v1/models.

    This is the fix for a whole class of failure that hit this project
    repeatedly: hardcoded model names 404'ing because Groq changed its
    catalog. Rather than guessing which model names are valid, ask the API.
    Returns [] on any failure, so callers fall back to the hardcoded list
    rather than breaking.
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None and not force_refresh:
        return _CATALOG_CACHE

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return []

    try:
        import requests

        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"[GroqClient] Could not fetch model catalog ({exc}). Using hardcoded list.")
        return []

    models = []
    for entry in payload.get("data", []):
        model_id = entry.get("id", "")
        if not model_id or entry.get("active") is False:
            continue
        if any(marker in model_id.lower() for marker in _NON_CHAT_MODEL_MARKERS):
            continue
        models.append(model_id)

    _CATALOG_CACHE = models
    print(f"[GroqClient] Live catalog: {len(models)} chat-capable models available.")
    return models


def _ordered_candidates(env_model: str) -> list[str]:
    """Builds the candidate list: env pin first, then catalog by preference."""
    candidates: list[str] = []
    if env_model:
        candidates.append(env_model)

    available = list_available_models()
    if available:
        # Sort the live catalog by our preference order; anything not in the
        # preference list goes last but is still kept as a genuine option.
        def rank(model_id: str) -> int:
            for i, prefix in enumerate(_MODEL_PREFERENCE):
                if model_id.startswith(prefix):
                    return i
            return len(_MODEL_PREFERENCE)

        for model_id in sorted(available, key=rank):
            if model_id not in candidates:
                candidates.append(model_id)
    else:
        for model_id in _FALLBACK_MODELS:
            if model_id not in candidates:
                candidates.append(model_id)

    return candidates


class _FallbackChatGroq:
    """
    Drop-in stand-in for a single ChatGroq instance that transparently
    advances through the model candidate list on a MODEL-LEVEL failure
    (404 model_not_found, decommissioned, access denied) — a different
    problem from rate limits, which invoke_with_retry (one layer up)
    already handles.

    WHY THIS EXISTS — a real bug this replaces:
    The original get_llm() built each candidate with `ChatGroq(model=m, ...)`
    inside a try/except, believing that loop was the fallback mechanism.
    But constructing a ChatGroq object NEVER contacts Groq's servers — it's
    local object construction, and can't fail for a validly-formatted model
    string. So that try/except could only ever catch a locally malformed
    call, never "this model doesn't exist" — that error only surfaces later,
    at .invoke() time, in a completely different function, by which point
    get_llm() had already returned and committed to the one broken model
    with no way back. Concretely: a Groq-side model deprecation (which is
    exactly what happened here — llama-3.3-70b-versatile returning 404)
    could never trigger the "5-model fallback chain" the code appeared to
    promise, because the failure and the fallback logic were in two
    different functions that never talked to each other.

    This class moves the fallback decision to where the real signal is: the
    first invoke() call. It's implemented here, not by changing
    invoke_with_retry or any of get_llm()'s call sites, so every existing
    caller keeps working unchanged — they all just call something with
    .invoke(), which is exactly what this still is.
    """

    # Deliberately narrow: matches Groq's actual model-unavailability wording
    # (seen in production: "does not exist or you do not have access to it",
    # code "model_not_found") plus the standard "decommissioned"/"deprecated"
    # phrasing providers use. NOT matched: 429/400/network errors — those are
    # different problems with different correct responses (retry-with-
    # backoff, or surface immediately), and conflating them with "try a
    # different model" would silently paper over real bugs by making them
    # look like a model-availability issue.
    _MODEL_UNAVAILABLE_PATTERN = re.compile(
        r"model_not_found|does not exist|do not have access|"
        r"has been decommissioned|has been deprecated|model.{0,20}unavailable",
        re.IGNORECASE,
    )

    # Groq rejects the request with this when the model emits a tool call
    # that the request's tool_choice/tools don't permit. Observed repeatedly
    # with gpt-oss models, including them calling their OWN built-in tools
    # (e.g. `web_open`) that this project never registered. It is NOT a
    # model-availability problem, so it gets its own recovery path: retry
    # once with tools stripped entirely from the request.
    _TOOL_USE_FAILED_PATTERN = re.compile(
        r"tool_use_failed|Tool choice is none, but model called a tool|"
        r"failed_generation",
        re.IGNORECASE,
    )

    def __init__(self, candidates, temperature, max_tokens, with_tools, tool_choice):
        self._candidates = list(candidates)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._with_tools = with_tools
        self._tool_choice = tool_choice
        self._tried: set[str] = set()
        self._current_model: Optional[str] = None
        self._current_llm = self._build(self._candidates[0])

    def _build(self, model: str):
        llm = ChatGroq(model=model, temperature=self._temperature, max_tokens=self._max_tokens)
        if self._with_tools:
            # tool_choice="none" is passed through to Groq as an EXPLICIT
            # constraint rather than being left implicit.
            #
            # Why this matters (a real production failure): the previous
            # approach to "stop the model calling tools on the final round"
            # was to simply not bind any tools. Per Groq's API docs,
            # tool_choice defaults to "none" when no tools are present — so
            # that *should* work. But gpt-oss models were observed emitting
            # tool calls anyway mid-conversation (including their OWN native
            # tools like `web_open`, which this project never registered),
            # and Groq's server-side validator then rejected the whole
            # request with 400 "Tool choice is none, but model called a
            # tool". Sending the tools list WITH an explicit
            # tool_choice="none" states the constraint directly instead of
            # relying on a server-side default inferred from an absent field.
            llm = llm.bind_tools(self._with_tools, tool_choice=self._tool_choice) \
                if self._tool_choice else llm.bind_tools(self._with_tools)
        elif self._tool_choice == "none":
            # tool_choice="none" with no tools to bind is a no-op — there's
            # nothing to forbid. Handled explicitly so it can't silently
            # become a malformed request.
            pass
        self._current_model = model
        return llm

    def bind_tools(self, *args, **kwargs):
        # get_llm() already binds tools before returning in the normal path;
        # this exists only so anything that calls .bind_tools() on the
        # returned object directly (rather than via get_llm's with_tools=)
        # still works, same as it would on a raw ChatGroq.
        self._current_llm = self._current_llm.bind_tools(*args, **kwargs)
        return self

    def invoke(self, messages):
        last_exc = None
        for model in self._candidates:
            if model in self._tried:
                continue
            if model != self._current_model:
                self._current_llm = self._build(model)
            try:
                response = self._current_llm.invoke(messages)
                if self._tried:
                    print(f"[GroqClient] Recovered using fallback model: {model}")
                return response
            except Exception as exc:
                error_text = str(exc)

                if self._TOOL_USE_FAILED_PATTERN.search(error_text):
                    # The model tried to call a tool it wasn't allowed to.
                    # Retry the SAME model once with tools removed from the
                    # request entirely — this is a request-shape problem, not
                    # a model-capability one, so swapping models wouldn't fix
                    # it and would just burn the candidate list.
                    print(
                        f"[GroqClient] '{model}' emitted a disallowed tool call. "
                        f"Retrying once with tools fully stripped..."
                    )
                    try:
                        bare = ChatGroq(
                            model=model,
                            temperature=self._temperature,
                            max_tokens=self._max_tokens,
                        )
                        return bare.invoke(messages)
                    except Exception as retry_exc:
                        print(f"[GroqClient] Tool-stripped retry also failed: {retry_exc}")
                        raise retry_exc from exc

                if self._MODEL_UNAVAILABLE_PATTERN.search(error_text):
                    print(
                        f"[GroqClient] Model '{model}' unavailable at invoke "
                        f"time ({exc}). Trying next candidate..."
                    )
                    self._tried.add(model)
                    last_exc = exc
                    continue
                # NOT a model-availability error (429, 400, network, etc.) —
                # exactly what invoke_with_retry one layer up is built to
                # handle. Let it propagate there unchanged rather than
                # masking a real error as "try a different model".
                raise

        raise RuntimeError(
            f"All Groq models unavailable. Last error: {last_exc}\n"
            f"Check your GROQ_API_KEY and current model availability at "
            f"https://console.groq.com/docs/models"
        )


def get_llm(
    temperature: float = 0.1,
    max_tokens: int = 4096,
    with_tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
) -> ChatGroq:
    """
    Returns a Groq LLM client using the best available model, with automatic
    fallback across the candidate list on a model-availability failure (not
    just at setup — see _FallbackChatGroq for why that distinction matters).

    Priority:
      1. GROQ_MODEL env var (set this in .env to pin a model)
      2. Auto-fallback through _FALLBACK_MODELS list

    Args:
        temperature:  LLM temperature (0.1 for reasoning, 0.3 for writing)
        max_tokens:   Max tokens for completion
        with_tools:   If provided, binds tools to the LLM (.bind_tools)
        tool_choice:  If provided along with with_tools, forces the model to
                      call exactly that named tool rather than choosing freely
                      (or "any"/"auto" if you want the model to still pick).
                      Used by the compute executor to force calculator calls.

    Usage:
        from utils.groq_client import get_llm
        llm = get_llm(temperature=0.2)
        llm_with_tools = get_llm(temperature=0.1, with_tools=ALL_TOOLS)
        forced = get_llm(with_tools=[calculator], tool_choice="calculator")
    """
    env_model  = os.getenv("GROQ_MODEL", "").strip()
    candidates = _ordered_candidates(env_model)

    if not candidates:
        raise RuntimeError(
            "[GroqClient] No Groq models available. Check GROQ_API_KEY is set "
            "and valid, then verify with: curl "
            "https://api.groq.com/openai/v1/models -H \"Authorization: Bearer "
            "$GROQ_API_KEY\""
        )

    print(f"[GroqClient] Primary candidate: {candidates[0]}")
    return _FallbackChatGroq(candidates, temperature, max_tokens, with_tools, tool_choice)


def invoke_with_retry(
    llm: ChatGroq,
    messages: list,
    max_retries: int = 5,
) -> object:
    """
    Invokes LLM with automatic retry on 429 rate-limit errors.
    Parses the wait time from Groq's error message and sleeps exactly that long.

    Args:
        llm:         ChatGroq instance (with or without tools bound)
        messages:    List of LangChain message objects
        max_retries: Max number of retry attempts before giving up

    Returns:
        LLM response object

    Raises:
        Exception: If all retries exhausted or non-rate-limit error occurs
    """
    import time
    import re

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)

        except Exception as exc:
            err_str  = str(exc)
            last_exc = exc

            if "429" in err_str or "rate_limit_exceeded" in err_str:
                # A TOKENS-PER-DAY limit and a tokens-PER-MINUTE limit both
                # surface as a 429, but they are NOT the same failure. A real
                # production incident: this code used to treat every 429
                # identically, parsing "try again in Xs" (which TPD errors
                # from Groq don't even include — they reset at a UTC day
                # boundary, not in seconds) and falling back to `attempt * 8`
                # seconds of backoff regardless. That meant retrying a
                # daily-exhausted quota 5 times with up to 65s of sleep each
                # — several minutes spent sleeping on a limit that CANNOT
                # reset within that window, guaranteed to fail anyway. Worse,
                # that sleep happens synchronously inside a single node call
                # with zero SSE events emitted, which is exactly the kind of
                # silence that gets an idle connection dropped client-side
                # ("Connection failed: network error") on a LATER, unrelated
                # request. Detecting "per day"/"TPD" and failing FAST with a
                # clear message is strictly better: same eventual outcome,
                # minutes faster, and the user knows to wait for tomorrow or
                # switch keys instead of watching a dead spinner.
                if re.search(r"per day|\btpd\b", err_str, re.IGNORECASE):
                    raise RuntimeError(
                        "[GroqClient] Daily token quota exhausted for this "
                        "model/API key. This resets on Groq's daily cycle, "
                        "not within seconds — retrying now cannot help. "
                        "Wait for the reset or switch to a different "
                        f"GROQ_API_KEY. Original error: {err_str}"
                    ) from exc

                # Parse exact wait time from Groq error message
                # e.g. "Please try again in 2.99s"
                wait_match = re.search(
                    r"try again in ([\d.]+)s", err_str, re.IGNORECASE
                )
                wait_secs = float(wait_match.group(1)) if wait_match else (attempt * 8)
                wait_secs = min(wait_secs + 2, 65)   # add 2s buffer, cap at 65s

                print(
                    f"[GroqClient] 429 rate limit (attempt {attempt}/{max_retries}). "
                    f"Sleeping {wait_secs:.0f}s..."
                )
                time.sleep(wait_secs)
                continue   # retry

            # Non-rate-limit error → raise immediately
            raise

    raise RuntimeError(
        f"[GroqClient] All {max_retries} retries exhausted. Last error: {last_exc}"
    )
