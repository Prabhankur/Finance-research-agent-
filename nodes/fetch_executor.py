"""
nodes/fetch_executor.py – Parallel-by-group fetch executor.

Instead of running all steps sequentially, groups steps by their tool's
rate-limit domain and runs independent groups concurrently:

  Group A (AV-paced): financial_data_api, earnings_transcript
    → Sequential within group (shared 5/min AV rate limit)
  Group B (SEC): sec_filing_search
    → Can run concurrently with other groups (SEC has its own rate limit)
  Group C (Web): web_search
    → Can run concurrently (Tavily, separate rate limit)
  Group D (FRED): fred_series
    → Can run concurrently (separate API)

Groups A, B, C, D all start simultaneously via ThreadPoolExecutor.
Within each group, steps run sequentially (to respect per-API limits).
Total time ≈ max(group_A_time, group_B_time, group_C_time, group_D_time)
instead of sum(all_steps).
"""
from __future__ import annotations
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from tools import FETCH_TOOL_FUNCTIONS, TOOL_DESCRIPTIONS
from utils.error_classifier import classify_result, is_failure, SYSTEMIC, TRANSIENT, BAD_ARGS
from utils.groq_client import get_llm, invoke_with_retry
from utils.helpers import safe_json_parse
from config import (
    LOCAL_RETRY_LIMIT, GLOBAL_REPLAN_BUDGET, COMPUTE_TOOLS, TRANSIENT_RETRY_PAUSE_S,
)

# Which tools share Alpha Vantage's rate limit
_AV_TOOLS = {"financial_data_api", "earnings_transcript"}
_SEC_TOOLS = {"sec_filing_search"}
_WEB_TOOLS = {"web_search"}
_FRED_TOOLS = {"fred_series"}


# An error is worth sending to the model for argument repair only if it reads
# like an argument problem. An unexpected bug (KeyError in a parser, say) lands
# in the same BAD_ARGS bucket by default, but no rewording of the arguments
# fixes it, so spending an LLM call on it would just waste the budget.
_ARGUMENT_HINT = re.compile(
    r"argument|parameter|required|invalid|must be|unknown|not a valid|expected|"
    r"unsupported|positive|missing|unexpected", re.I,
)

_ARG_FIX_PROMPT = f"""You repair ONE malformed tool call in a financial research
pipeline. You are given the tool name, the arguments that were sent, and the
error the tool returned. Reply with ONLY a JSON object holding the corrected
arguments — the same keys the tool accepts, no commentary, no markdown fences.
If the error cannot be fixed by changing the arguments, reply with {{}}.

{TOOL_DESCRIPTIONS}"""


def _correct_args(tool_name: str, args: dict, error_text: str) -> Optional[dict]:
    """
    One model call to repair a bad-arguments failure — the "someone has to
    reason" branch of the failure policy. Retrying the SAME arguments (what this
    step used to do) can only reproduce the same error, because the plan fixed
    them in advance; changing them needs judgement.

    Returns corrected arguments, or None when there is nothing useful to retry
    (model unavailable, unparseable reply, empty object, or arguments unchanged).
    """
    if not _ARGUMENT_HINT.search(error_text or ""):
        return None
    try:
        llm = get_llm(temperature=0.0, max_tokens=300)
        response = invoke_with_retry(llm, [
            SystemMessage(content=_ARG_FIX_PROMPT),
            HumanMessage(content=(
                f"Tool: {tool_name}\nArguments sent: {args}\nError: {error_text[:600]}"
            )),
        ])
        fixed = safe_json_parse(str(response.content or ""))
    except Exception:
        return None
    if isinstance(fixed, dict) and fixed and fixed != args:
        return fixed
    return None


def _run_one_fetch_step(step: dict, disabled_tools: set) -> dict:
    """Executes a single plan step under the failure policy:
    TRANSIENT -> one retry of the same call after a pause (no model needed);
    BAD_ARGS  -> one retry with arguments repaired by the model (a same-args
                 retry cannot succeed); SYSTEMIC -> no retry, the caller
                 disables the tool and replans; empty -> reported as such."""
    tool_name = step.get("tool")
    args = dict(step.get("args", {}))
    step_num = step.get("step")

    if tool_name in disabled_tools:
        return {
            "step": step_num, "tool": tool_name, "status": "skipped",
            "content": f"[Skipped] '{tool_name}' was disabled earlier this run.",
            "retries_used": 0, "error_category": None,
        }

    fn = FETCH_TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return {
            "step": step_num, "tool": tool_name, "status": "skipped",
            "content": f"[Skipped] Unknown fetch tool '{tool_name}'.",
            "retries_used": 0, "error_category": None,
        }

    retries_used = 0
    result = None
    category = None
    for attempt in range(LOCAL_RETRY_LIMIT + 1):
        try:
            result = fn(**args)
        except Exception as exc:
            result = f"[Error] {tool_name} raised an exception: {exc}"
        category = classify_result(tool_name, result)

        if category == SYSTEMIC:
            break
        if not is_failure(category):
            break
        if attempt >= LOCAL_RETRY_LIMIT:
            break                               # retry budget spent

        if category == TRANSIENT:
            # Same call again, but not instantly: an immediate retry meets the
            # same blip (it used to run with a 0.00s gap).
            time.sleep(TRANSIENT_RETRY_PAUSE_S)
        elif category == BAD_ARGS:
            corrected = _correct_args(tool_name, args, result)
            if corrected is None:
                break                           # nothing a retry could change
            args = corrected
        retries_used = attempt + 1

    status = "success" if not is_failure(category) else (
        "systemic_failure" if category == SYSTEMIC else "failed"
    )
    return {
        "step": step_num, "tool": tool_name, "status": status,
        "content": result, "retries_used": retries_used,
        "error_category": category if is_failure(category) else None,
    }


def _run_group(steps: list[dict], disabled_tools: set, group_name: str) -> list[dict]:
    """Run a group of steps sequentially (they share a rate limit)."""
    results = []
    for step in steps:
        entry = _run_one_fetch_step(step, disabled_tools)
        print(f"[FetchExecutor/{group_name}] step {entry['step']} · {entry['tool']} · {entry['status']}"
              + (f" (retries: {entry['retries_used']})" if entry["retries_used"] else "")
              + (f"\n    → {str(entry['content'])[:200]}" if entry["status"] != "success" else ""))
        results.append(entry)

        # If systemic failure, stop this group (but other groups continue)
        if entry["status"] == "systemic_failure":
            break
    return results


def fetch_executor_node(state: AgentState) -> dict:
    plan = state.get("roadmap", {}).get("research_plan", [])
    disabled_tools = set(state.get("disabled_tools", []))
    replan_count = state.get("replan_count", 0)

    existing_chunks = list(state.get("context_chunks", []))
    existing_calls = list(state.get("tool_calls_made", []))
    already_done_steps = {c["step"] for c in existing_chunks if c.get("status") == "success"}

    fetch_steps = [
        s for s in plan
        if s.get("tool") not in COMPUTE_TOOLS and s.get("step") not in already_done_steps
    ]

    # ── Group steps by rate-limit domain ──────────────────────────────────────
    av_steps = [s for s in fetch_steps if s.get("tool") in _AV_TOOLS]
    sec_steps = [s for s in fetch_steps if s.get("tool") in _SEC_TOOLS]
    web_steps = [s for s in fetch_steps if s.get("tool") in _WEB_TOOLS]
    fred_steps = [s for s in fetch_steps if s.get("tool") in _FRED_TOOLS]
    other_steps = [s for s in fetch_steps if s.get("tool") not in
                   _AV_TOOLS | _SEC_TOOLS | _WEB_TOOLS | _FRED_TOOLS]

    groups = []
    if av_steps:    groups.append((av_steps, "AV"))
    if sec_steps:   groups.append((sec_steps, "SEC"))
    if web_steps:   groups.append((web_steps, "WEB"))
    if fred_steps:  groups.append((fred_steps, "FRED"))
    if other_steps: groups.append((other_steps, "OTHER"))

    print(f"[FetchExecutor] {len(fetch_steps)} steps in {len(groups)} parallel groups: "
          + ", ".join(f"{name}({len(steps)})" for steps, name in groups))

    # ── Run groups in parallel ────────────────────────────────────────────────
    all_results: list[dict] = []

    if len(groups) <= 1:
        # Single group or empty — no need for threads
        for steps, name in groups:
            all_results.extend(_run_group(steps, disabled_tools, name))
    else:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(_run_group, steps, disabled_tools, name): name
                for steps, name in groups
            }
            for future in as_completed(futures):
                group_name = futures[future]
                try:
                    group_results = future.result()
                    all_results.extend(group_results)
                except Exception as exc:
                    print(f"[FetchExecutor] Group {group_name} crashed: {exc}")

    # ── Process results ───────────────────────────────────────────────────────
    needs_replan = False
    failed_step_for_replan = None

    for entry in all_results:
        existing_chunks.append(entry)
        existing_calls.append({
            "step": entry["step"], "tool": entry["tool"],
            "status": entry["status"], "retries_used": entry["retries_used"],
        })

        if entry["status"] == "systemic_failure":
            disabled_tools.add(entry["tool"])
            if replan_count < GLOBAL_REPLAN_BUDGET and not needs_replan:
                needs_replan = True
                failed_step_for_replan = entry

    return {
        "context_chunks": existing_chunks,
        "tool_calls_made": existing_calls,
        "disabled_tools": list(disabled_tools),
        "needs_replan": needs_replan,
        "replan_reason": failed_step_for_replan,
        "replan_phase": "fetch" if needs_replan else None,
    }
