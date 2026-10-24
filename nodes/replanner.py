"""
nodes/replanner.py – Shared recovery node, reachable from both the fetch
executor and the compute executor when either hits a systemic failure that
survived its local retry budget.

Only reached for SYSTEMIC failures (a tool is broken for the rest of this
run) — bad-args and transient failures are handled by local retry inside the
executors and never get here. That split matters: a full replan call is
relatively expensive (LLM call, has to see the whole remaining plan), so we
reserve it for the one failure type where a same-tool retry genuinely can't
help, and where later steps referencing the now-dead tool need rewriting,
not just the one that failed.

Bounded by a GLOBAL budget (config.GLOBAL_REPLAN_BUDGET), shared across both
executors, not per-step — this is what stops a systemic outage from causing
an endless replan loop.
"""
from __future__ import annotations
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from tools import TOOL_DESCRIPTIONS
from utils.groq_client import get_llm, invoke_with_retry
from utils.helpers import safe_json_parse

REPLANNER_SYSTEM_PROMPT = f"""You are recovering a financial research plan after
a tool call failed for a systemic reason (the tool itself is broken for the
rest of this run — not just a bad argument).

You will be shown:
  - the original research goal
  - which tools are now permanently unavailable this run (disabled_tools)
  - the steps already completed successfully (do not repeat these)
  - the full list of steps that were still remaining before the failure
  - the specific step that failed and why

{TOOL_DESCRIPTIONS}

Produce a revised list of REMAINING steps only (do not include already
completed steps). For every remaining step that referenced a disabled tool,
either:
  - route it to a different available tool that can get equivalent or
    partial data, or
  - drop it and note in "notes" that this data will be unavailable this run.
Steps that don't touch a disabled tool can be carried over unchanged.

OUTPUT FORMAT: Respond ONLY with valid JSON, no markdown fences.
{{
  "remaining_steps": [
    {{"step": 6, "tool": "web_search",
      "args": {{"query": "AAPL Q3 2024 revenue quarterly report"}},
      "purpose": "Substitute for blocked 10-Q pull", "notes": "..."}}
  ],
  "unavailable_data_notes": ["10-Q structured filing data unavailable this run: edgartools dependency missing"]
}}"""


def _fallback_revised_plan(remaining_steps: list[dict], disabled_tools: set[str]) -> dict:
    """If the replanner LLM call itself fails to parse, degrade safely:
    drop any step touching a disabled tool rather than looping on it again."""
    kept = [s for s in remaining_steps if s.get("tool") not in disabled_tools]
    dropped = [s for s in remaining_steps if s.get("tool") in disabled_tools]
    notes = [
        f"Step {s.get('step')} ({s.get('purpose', '')}) dropped: tool '{s.get('tool')}' unavailable this run."
        for s in dropped
    ]
    return {"remaining_steps": kept, "unavailable_data_notes": notes}


def replanner_node(state: AgentState) -> dict:
    plan = state.get("roadmap", {}).get("research_plan", [])
    disabled_tools = set(state.get("disabled_tools", []))
    replan_count = state.get("replan_count", 0)
    failed_entry = state.get("replan_reason") or {}
    completed_steps = {
        c["step"] for c in state.get("context_chunks", []) if c.get("status") == "success"
    }

    remaining_steps = [s for s in plan if s.get("step") not in completed_steps]

    llm = get_llm(temperature=0.1, max_tokens=2048)
    user_content = (
        f"ORIGINAL GOAL: {state.get('elaborated_query', state.get('raw_query', ''))}\n\n"
        f"DISABLED TOOLS THIS RUN: {sorted(disabled_tools)}\n\n"
        f"COMPLETED STEPS (do not repeat): {sorted(completed_steps)}\n\n"
        f"REMAINING STEPS BEFORE FAILURE:\n{json.dumps(remaining_steps, indent=2)}\n\n"
        f"FAILED STEP: {json.dumps(failed_entry, default=str)}\n"
    )
    messages = [SystemMessage(content=REPLANNER_SYSTEM_PROMPT), HumanMessage(content=user_content)]

    try:
        response = invoke_with_retry(llm, messages)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.content.strip(), flags=re.MULTILINE)
        revised = safe_json_parse(raw, fallback=None)
        if revised is None or "remaining_steps" not in revised:
            revised = _fallback_revised_plan(remaining_steps, disabled_tools)
    except Exception as exc:
        print(f"[Replanner] LLM call failed ({exc}); using safe fallback (drop disabled-tool steps).")
        revised = _fallback_revised_plan(remaining_steps, disabled_tools)

    new_remaining = revised.get("remaining_steps", [])
    notes = revised.get("unavailable_data_notes", [])

    # Rebuild the plan: completed steps stay as-is (for provenance), replaced
    # with the revised remaining steps going forward.
    kept_completed = [s for s in plan if s.get("step") in completed_steps]
    new_plan = kept_completed + new_remaining

    roadmap = dict(state.get("roadmap", {}))
    roadmap["research_plan"] = new_plan
    existing_notes = roadmap.get("data_limitation_notes", [])
    roadmap["data_limitation_notes"] = existing_notes + notes

    print(f"[Replanner] Revised plan: {len(new_remaining)} remaining steps "
          f"(replan {replan_count + 1} used this run). Notes: {notes}")

    return {
        "roadmap": roadmap,
        "replan_count": replan_count + 1,
        "needs_replan": False,
        "replan_reason": None,
        # replan_phase intentionally left as-is in state — the conditional
        # edge reads it to know whether to route back to fetch or compute.
    }
