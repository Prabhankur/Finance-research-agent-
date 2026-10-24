"""
evaluation/evaluator.py  –  LLM-as-judge scoring for real conversation turns.

Every completed chat turn produces a scored record: on 5 rubric dimensions,
each 0-100. Records land in the evaluation_scores table (see
database/schema.py augmentation below) and roll up into a periodic
"what to improve" report — the "self-evaluation" piece you asked for.

DESIGN DECISIONS THAT MATTER FOR THE DEFENSE

1. **The rubric is fixed and public.** A judge is only trustworthy if its
   criteria don't drift. The 5 rubric dimensions and their guidance below
   are the whole rubric — the LLM is not asked to invent criteria.

2. **The judge is a different, cheaper LLM call than synthesis.** Using
   the same model to grade its own work is a well-known failure pattern
   (a model can't reliably detect its own hallucinations). We prompt it
   with the OBJECTIVE facts (what tools returned, what the DB says, what
   contradictions were detected) alongside the answer, so its judgement
   is grounded in checkable data rather than vibes.

3. **Failure returns None-scores, not zero-scores.** A score of 0 means
   "the answer was actively bad on this dimension"; a score of None means
   "we couldn't score this dimension". The rollup treats them differently —
   Nones don't drag averages down.

4. **The rollup is deterministic aggregation**, not another LLM call. Once
   you have scored records in a table, "which dimensions are trending
   worst" is a SQL question, not a language question.

5. **Runs OUT OF BAND from the user turn.** The user shouldn't wait for
   grading — it happens on a worker thread after the SSE stream closes.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from database.connection import execute, init_db, query, query_one


# ─────────────────────────────────────────────────────────────────────────────
#  Schema — evaluation_scores lives outside the financial-facts schema
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id           TEXT,
    turn_id             TEXT NOT NULL,
    query               TEXT NOT NULL,
    answer              TEXT NOT NULL,
    turn_status         TEXT,
    -- 5 rubric dimensions, 0-100 each, NULL if this dim couldn't be scored
    grounding_score     INTEGER,
    citation_score      INTEGER,
    reconciliation_score INTEGER,
    completeness_score  INTEGER,
    format_score        INTEGER,
    -- Deterministic, NOT LLM-judged — see check_calculation_fidelity below.
    -- NULL when no calculator calls happened this turn (not applicable),
    -- same convention as reconciliation_score when there were no
    -- contradictions to check.
    calculation_fidelity_score INTEGER,
    overall_score       INTEGER,
    -- One-line human-readable summary from the judge per turn
    judge_summary       TEXT,
    -- Structured metadata about the turn — tool count, contradictions seen,
    -- token cost — for the rollup query. JSON blob keeps the schema stable.
    turn_metadata       TEXT,
    created_at          TEXT NOT NULL
)
"""
_EVAL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_eval_thread ON evaluation_scores(thread_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluation_scores(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_eval_overall ON evaluation_scores(overall_score)",
]


def _init() -> None:
    init_db()
    try:
        from database.connection import _translate, get_connection
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(_translate(_EVAL_SCHEMA))
            for stmt in _EVAL_INDEXES:
                cursor.execute(_translate(stmt))

            # Migration for DBs created before calculation_fidelity_score
            # existed: CREATE TABLE IF NOT EXISTS does nothing for an
            # already-existing table, so a fresh column needs an explicit
            # ADD COLUMN, guarded because it errors if the column is
            # already there (no portable "ADD COLUMN IF NOT EXISTS" across
            # both SQLite and Postgres syntax in one statement).
            try:
                cursor.execute(_translate(
                    "ALTER TABLE evaluation_scores ADD COLUMN calculation_fidelity_score INTEGER"
                ))
            except Exception:
                pass  # column already exists — expected on every run after the first
    except Exception as exc:
        print(f"[Evaluator] Schema init failed ({exc}) — running without persisted scores.")


_init()


# ─────────────────────────────────────────────────────────────────────────────
#  Rubric — defended as fixed and observable
# ─────────────────────────────────────────────────────────────────────────────

RUBRIC = {
    "grounding": (
        "Are all specific numbers in the answer traceable to a tool result "
        "or the retrieved-context section? A number that appears in the "
        "answer but NOT in the tool results is a hallucination and should "
        "score low. Prose without numbers (explanation, framing) is fine."
    ),
    "citation": (
        "Is every non-trivial figure attributed to its source, ideally "
        "with the tier ('yfinance, tier B' etc)? Bare numbers with no "
        "source citation score low, even if they are correct."
    ),
    "reconciliation": (
        "If the retrieved data contained a cross-source contradiction "
        "(listed below), did the answer surface it to the user — leading "
        "with the higher-tier source and disclosing the delta — or did it "
        "silently pick one? If no contradictions were flagged, this "
        "dimension is not applicable (score null)."
    ),
    "completeness": (
        "Does the answer address every part of the user's question? A "
        "multi-part question ('compare X, Y, Z on revenue and margin') "
        "that only covers X's revenue scores low."
    ),
    "format": (
        "Is the answer conversational, appropriately concise, with a table "
        "when comparing multiple values, and free of scaffolding preamble "
        "('Great question!') or a closing sales pitch? Prose that reads "
        "like a document dump when a chat answer was asked for scores low."
    ),
}


_JUDGE_PROMPT_TEMPLATE = """You are evaluating a financial research assistant's
answer. You are NOT asked to answer the question yourself, only to score how
well the assistant did.

Score on 5 dimensions, 0-100 each. Return JSON only — no prose, no code
fences, no explanation outside the JSON structure:

{{
  "grounding_score":     <int 0-100 or null>,
  "citation_score":      <int 0-100 or null>,
  "reconciliation_score": <int 0-100 or null>,
  "completeness_score":  <int 0-100 or null>,
  "format_score":        <int 0-100 or null>,
  "overall_score":       <int 0-100>,
  "summary": "<one line, under 25 words, describing the main weakness or 'no notable issues'>"
}}

Use null for reconciliation_score ONLY if the "Contradictions flagged"
section below is empty. Otherwise it MUST be a real number.

## Rubric

grounding: {grounding}

citation: {citation}

reconciliation: {reconciliation}

completeness: {completeness}

format: {format}

## User's question

{question}

## Assistant's answer

{answer}

## Tool results the assistant had access to

{tool_context}

## Contradictions flagged by the deterministic detector

{contradictions}

Return only the JSON object described above.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_tool_results(successful_results: list[dict], max_chars: int = 2000) -> str:
    """Compresses tool payloads so the judge prompt stays within a sane
    token budget — the judge doesn't need the full 10-K text, only the
    figures the assistant could have used."""
    if not successful_results:
        return "(no tool results — the assistant answered from prior context or general knowledge)"

    parts: list[str] = []
    remaining = max_chars
    for record in successful_results:
        header = f"[{record.get('tool', '?')}] args={json.dumps(record.get('args') or {})}"
        body = str(record.get("result") or "")[: max(200, remaining // len(successful_results))]
        block = f"{header}\n{body}"
        parts.append(block)
        remaining -= len(block)
        if remaining <= 0:
            break

    return "\n\n".join(parts)


def _summarise_contradictions(contradictions: list[dict]) -> str:
    if not contradictions:
        return "(none)"
    lines = []
    for note in contradictions:
        current = note.get("current", {})
        alts = note.get("alternates", [])
        alt_str = ", ".join(
            f"{a.get('provider')} says {a.get('value')} (tier {a.get('tier')})"
            for a in alts
        )
        period = f" {note.get('period_end')}" if note.get('period_end') else ""
        lines.append(
            f"- {note.get('ticker')} {note.get('field')}{period}: "
            f"{current.get('provider')} says {current.get('value')} "
            f"(tier {current.get('tier')}); {alt_str}"
        )
    return "\n".join(lines)


def _parse_judge_response(raw: str) -> dict:
    """Extracts JSON from the judge's response, tolerant of stray text."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip().rstrip("`").strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in judge response")
    return json.loads(stripped[start : end + 1])


def _clamp_score(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(100, score))


def _extract_numbers(text: str) -> set[str]:
    """Pulls numeric tokens out of text for fidelity comparison. Normalized
    to a comparable string form (strip $ , % commas, keep 2 decimal places
    where present) so '16.35%' in the answer matches '16.35' from the
    calculator's raw output."""
    import re as _re
    raw = _re.findall(r"-?\d[\d,]*\.?\d*", text)
    normalized = set()
    for token in raw:
        cleaned = token.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        # Round to 2 decimals for comparison — protects against a case where
        # the calculator returns 16.3500 and the answer writes 16.35, which
        # are the SAME number, not a fidelity failure.
        normalized.add(f"{value:.2f}")
    return normalized


def check_calculation_fidelity(
    successful_results: list[dict], answer: str
) -> Optional[dict]:
    """
    Deterministic (NOT LLM) check: for every successful `calculator` call
    this turn, does the number it actually returned appear somewhere in the
    final written answer?

    WHY THIS IS DIFFERENT FROM THE LLM-JUDGE DIMENSIONS
    The calculator is pure Python arithmetic — for a calculated metric,
    there IS a ground truth, unlike a raw fetched figure where two sources
    can legitimately disagree. So this doesn't need a judge's opinion; it's
    a string/number match. This directly answers the "the evaluator should
    at least know the correct answer" concern raised about the other
    dimensions — for CALCULATED metrics specifically, it now does.

    Returns None if no calculator calls happened this turn (dimension is
    not applicable — same convention as reconciliation_score when there
    were no contradictions), else a dict with the score and any mismatches
    found, for storage and for the rollup to surface.
    """
    calculator_results = [
        r for r in successful_results if r.get("tool") == "calculator"
    ]
    if not calculator_results:
        return None

    answer_numbers = _extract_numbers(answer)
    mismatches: list[dict] = []
    checked = 0

    for record in calculator_results:
        result_text = str(record.get("result") or "")
        calculator_numbers = _extract_numbers(result_text)
        if not calculator_numbers:
            continue
        checked += 1
        # A calculator call is "reflected" if AT LEAST ONE of its numbers
        # made it into the answer. Calculator results often carry more than
        # one number (e.g. formula + result), and the answer only needs to
        # state the actual result, not restate every intermediate figure.
        if not (calculator_numbers & answer_numbers):
            mismatches.append({
                "args": record.get("args"),
                "calculator_said": result_text[:200],
            })

    if checked == 0:
        return None

    score = round(100 * (checked - len(mismatches)) / checked)
    return {
        "score": score,
        "checked": checked,
        "mismatches": mismatches,
    }


def score_turn(
    *,
    turn_id: str,
    thread_id: Optional[str],
    question: str,
    answer: str,
    successful_tool_results: list[dict],
    contradictions: list[dict],
    turn_status: str,
) -> Optional[dict]:
    """
    Runs the judge on one turn and persists the result.

    Returns the parsed judge output, or None on any failure. This function
    itself never raises — evaluation is a nice-to-have quality signal, not
    a load-bearing part of the request path.
    """
    # Import lazily so the evaluator module has no import-time cost for
    # users who don't call it (the report pipeline imports lots of heavy
    # tools, and this module lives in database.connection's neighbourhood).
    from utils.groq_client import get_llm, invoke_with_retry
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        grounding=RUBRIC["grounding"],
        citation=RUBRIC["citation"],
        reconciliation=RUBRIC["reconciliation"],
        completeness=RUBRIC["completeness"],
        format=RUBRIC["format"],
        question=question[:1500],
        answer=answer[:3000],
        tool_context=_summarise_tool_results(successful_tool_results),
        contradictions=_summarise_contradictions(contradictions),
    )

    try:
        # A separate, cheap model — judgment doesn't need the same class of
        # reasoning as synthesis. temperature=0 for determinism: the same
        # turn scored twice should produce the same score.
        llm = get_llm(temperature=0.0, max_tokens=400)
        response = invoke_with_retry(
            llm,
            [
                SystemMessage(content="You return only JSON, no prose."),
                HumanMessage(content=prompt),
            ],
        )
        parsed = _parse_judge_response(str(response.content))
    except Exception as exc:
        print(f"[Evaluator] Judge call failed ({exc}) — turn not scored.")
        return None

    scores = {
        "grounding_score":     _clamp_score(parsed.get("grounding_score")),
        "citation_score":      _clamp_score(parsed.get("citation_score")),
        "reconciliation_score": _clamp_score(parsed.get("reconciliation_score")),
        "completeness_score":  _clamp_score(parsed.get("completeness_score")),
        "format_score":        _clamp_score(parsed.get("format_score")),
        "overall_score":       _clamp_score(parsed.get("overall_score")),
        "summary":             str(parsed.get("summary") or "").strip()[:400],
    }

    # Deterministic check — computed here, not by the LLM judge, and not
    # part of the judge prompt at all. This is what closes the "the
    # evaluator should at least know the correct answer" gap for CALCULATED
    # metrics: the calculator's own output IS the ground truth here.
    fidelity = check_calculation_fidelity(successful_tool_results, answer)
    fidelity_score = fidelity["score"] if fidelity else None
    if fidelity and fidelity["mismatches"]:
        print(
            f"[Evaluator] Calculation fidelity: {len(fidelity['mismatches'])} "
            f"calculator result(s) not reflected in the final answer."
        )

    metadata = {
        "tool_count":            len(successful_tool_results),
        "contradiction_count":   len(contradictions),
        "turn_status":           turn_status,
        "fidelity_mismatches":   fidelity["mismatches"] if fidelity else [],
    }

    try:
        execute(
            """
            INSERT INTO evaluation_scores
                (thread_id, turn_id, query, answer, turn_status,
                 grounding_score, citation_score, reconciliation_score,
                 completeness_score, format_score, calculation_fidelity_score,
                 overall_score, judge_summary, turn_metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id, turn_id, question[:2000], answer[:5000], turn_status,
                scores["grounding_score"], scores["citation_score"],
                scores["reconciliation_score"], scores["completeness_score"],
                scores["format_score"], fidelity_score, scores["overall_score"],
                scores["summary"], json.dumps(metadata),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    except Exception as exc:
        print(f"[Evaluator] Score persistence failed ({exc}).")

    scores["calculation_fidelity_score"] = fidelity_score
    return scores


def score_turn_async(**kwargs) -> None:
    """Fire-and-forget scoring — used from the API layer so the SSE stream
    isn't held open while the judge thinks."""
    thread = threading.Thread(target=score_turn, kwargs=kwargs, daemon=True)
    thread.start()


# ─────────────────────────────────────────────────────────────────────────────
#  Rollup — the "what should we improve" report
# ─────────────────────────────────────────────────────────────────────────────

def rollup_report(days: int = 7) -> dict[str, Any]:
    """
    Deterministic aggregation across recent scored turns. Answers questions
    like "which dimensions are trending worst" without another LLM call.

    Structure:
        {
            "window_days":       int,
            "n_turns_scored":    int,
            "avg_scores":        {dim: float | None},
            "n_below_50":        {dim: int},
            "recent_low_scoring": [{turn_id, query, overall, summary}, ...],
            "worst_dimension":   str | None,
        }
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        rows = query(
            """
            SELECT * FROM evaluation_scores
            WHERE created_at > ?
            ORDER BY created_at DESC
            """,
            (cutoff,),
        )
    except Exception as exc:
        return {"error": str(exc), "window_days": days}

    if not rows:
        return {
            "window_days": days, "n_turns_scored": 0,
            "avg_scores": {}, "n_below_50": {},
            "recent_low_scoring": [], "worst_dimension": None,
            "message": "No scored turns yet in this window.",
        }

    dims = ["grounding_score", "citation_score", "reconciliation_score",
            "completeness_score", "format_score", "calculation_fidelity_score"]

    avg_scores: dict[str, Optional[float]] = {}
    n_below_50: dict[str, int] = {}
    for dim in dims:
        values = [r[dim] for r in rows if r[dim] is not None]
        avg_scores[dim] = (sum(values) / len(values)) if values else None
        n_below_50[dim] = sum(1 for v in values if v < 50)

    # "Worst dimension" is the one where a MEANINGFUL fraction of turns
    # scored below 50 — not the one with the lowest average, because a
    # single 0-scored outlier can distort an average without indicating a
    # real pattern.
    if rows:
        worst = max(
            n_below_50.items(),
            key=lambda kv: (kv[1] / len(rows), -avg_scores.get(kv[0], 100) if avg_scores.get(kv[0]) is not None else 0),
        )
        worst_dimension = worst[0] if worst[1] > 0 else None
    else:
        worst_dimension = None

    low_scoring = [
        {
            "turn_id":       r["turn_id"],
            "query":         (r["query"] or "")[:200],
            "overall_score": r["overall_score"],
            "summary":       r["judge_summary"],
            "created_at":    r["created_at"],
        }
        for r in rows
        if r["overall_score"] is not None and r["overall_score"] < 60
    ][:20]

    return {
        "window_days": days,
        "n_turns_scored": len(rows),
        "avg_scores": {k: (round(v, 1) if v is not None else None) for k, v in avg_scores.items()},
        "n_below_50": n_below_50,
        "recent_low_scoring": low_scoring,
        "worst_dimension": worst_dimension,
    }


def recent_scores(limit: int = 20) -> list[dict]:
    """Recent scored turns — powers a debug endpoint and the eventual UI panel."""
    try:
        rows = query(
            """SELECT id, turn_id, thread_id, query, overall_score,
                      grounding_score, citation_score, reconciliation_score,
                      completeness_score, format_score, calculation_fidelity_score,
                      judge_summary, created_at
               FROM evaluation_scores ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        return rows
    except Exception:
        return []


def scores_for_thread(thread_id: str, limit: int = 100) -> list[dict]:
    """
    Every scored turn from ONE conversation, oldest first — the thing this
    module's schema always supported (thread_id has been a column and an
    index since the table was created) but had no dedicated query for.
    Ordered chronologically (not newest-first like recent_scores) because
    the natural use of this is "how did quality trend across this specific
    conversation", which reads more naturally start to end.
    """
    try:
        return query(
            """SELECT id, turn_id, query, overall_score, grounding_score,
                      citation_score, reconciliation_score, completeness_score,
                      format_score, calculation_fidelity_score, judge_summary,
                      created_at
               FROM evaluation_scores
               WHERE thread_id = ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (thread_id, limit),
        )
    except Exception:
        return []
