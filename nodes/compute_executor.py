"""
nodes/compute_executor.py – Executes "compute" steps (currently just
calculator; any future derived-metric tool added to config.COMPUTE_TOOLS
lands here too).

Unlike fetch tools, a compute tool's numeric inputs usually don't exist at
plan time — e.g. calculator needs AAPL's actual revenue figures, which only
exist after financial_data_api has run. So this step genuinely needs an
LLM to read the accumulated context and fill in the right numbers — that's
real, necessary agentic work, not redundant re-deciding.

To remove the OTHER kind of drift risk (the model calling a different tool,
or no tool at all), tool_choice is forced to the exact planned tool name —
the model can only fill in arguments, it cannot pick something else or
skip the call.

Same shared error classifier as the fetch executor: bad-args gets a local
retry (this is the common case here — a malformed calculator call), while a
genuinely unavailable input (e.g. the number it needs was never fetched
because that source got blacklisted) also just gets tagged and left for the
replanner to decide "this derived metric isn't computable, drop it" —
no new mechanism, same escalation path as fetch tools.
"""
from __future__ import annotations
from __future__ import annotations
import json
import re
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from graph.state import AgentState
from tools import TOOLS_BY_NAME, TOOL_DESCRIPTIONS
from utils.groq_client import get_llm, invoke_with_retry
from utils.error_classifier import classify_result, is_failure, SYSTEMIC
from config import LOCAL_RETRY_LIMIT, GLOBAL_REPLAN_BUDGET, COMPUTE_TOOLS, COMPUTE_STEP_CAP

COMPUTE_SYSTEM_PROMPT = f"""You compute one financial metric using data already
gathered this run. You will be told which metric to compute and shown the
research context collected so far. Find the exact numeric inputs that metric
needs in that context and call the tool with real numbers — never invent a
number that isn't in the context.

{TOOL_DESCRIPTIONS}

WHERE TO FIND NUMBERS IN THE CONTEXT:

The context contains data blocks tagged like [FinData] Market Overview,
[FinData] Income Statement, [FinData] Balance Sheet, [FinData] Cash Flow,
and [FinData] Key Ratios. Here's how to map calculator arguments to
context fields:

  For CAGR (start_value, end_value, years):
    → Look in [FinData] Income Statement for "Total Revenue" per year
    → start_value = earliest year's revenue, end_value = latest year's revenue
    → years = number of periods between them (e.g. 2022→2024 = 2 years)

  For ROE (net_income, shareholder_equity):
    → net_income: [FinData] Income Statement "Net Income" (latest year)
       OR [FinData] Market Overview "Net Income (TTM)"
    → shareholder_equity: [FinData] Balance Sheet "Shareholder Equity" (latest)

  For Free Cash Flow (operating_cash_flow, capex):
    → [FinData] Cash Flow "Operating Cash Flow" and "CapEx" (same period)

  For FCF Yield (fcf, market_cap):
    → fcf: computed as operating_cash_flow - capex from Cash Flow block
    → market_cap: [FinData] Market Overview "Market Cap"

  For Gross Margin (revenue, AND EITHER cogs OR gross_profit):
    → revenue: [FinData] Income Statement "Total Revenue"
    → gross_profit: [FinData] Income Statement "Gross Profit" — pass this
       DIRECTLY as gross_profit if it's present. Do NOT subtract it from
       revenue to compute cogs yourself — the calculator now accepts
       gross_profit as-is (see below). Only fall back to cogs if gross_profit
       genuinely isn't in the context anywhere.

  For Debt-to-Equity (total_debt, shareholder_equity):
    → total_debt: [FinData] Balance Sheet "Total Debt"
    → shareholder_equity: [FinData] Balance Sheet "Shareholder Equity"

CRITICAL RULES:
1. Your ONLY output should be one calculator tool call. No prose. No explanation.
2. Do NOT call any tool other than calculator — you only have access to
   calculator in this step. If you think you need to fetch more data, you
   cannot — just leave the missing argument out and the executor will handle it.
3. If a required numeric input genuinely isn't present anywhere in the context
   (not just hard to find — actually absent), call the tool anyway with whatever
   arguments you CAN fill. Leave missing arguments out entirely — do NOT
   substitute zero, a guess, a peer's number, or an "industry average."
4. Do NOT compute a number that already exists in the context. If the context
   already contains "Return on Equity (%): 148.75" from [FinData], that IS the
   ROE — calling calculator to re-derive it is redundant and wastes a step.
5. TTM (trailing twelve months) values from Market Overview are acceptable
   substitutes when annual statement data isn't available."""


# Metrics that can be read directly from fetched context instead of calculated.
# Maps metric name → list of text patterns that indicate the value already exists.
_METRIC_ALREADY_IN_CONTEXT = {
    "roe":              ["Return on Equity", "ROE (%)"],
    "gross_margin":     ["Gross Margin", "grossProfit"],
    "revenue_growth":   ["Quarterly Revenue Growth", "Revenue Growth"],
    "debt_to_equity":   ["Debt/Equity"],
    "current_ratio":    ["Current Ratio"],
    "interest_coverage": ["Interest Coverage"],
    "peg_ratio":        ["PEG Ratio"],
    "enterprise_value": ["Enterprise Value", "enterpriseValue"],
}


def _metric_already_available(metric: str, context_text: str) -> Optional[str]:
    """If the metric already exists in the fetched context, return the
    relevant snippet. Otherwise return None."""
    patterns = _METRIC_ALREADY_IN_CONTEXT.get(metric.lower(), [])
    for pat in patterns:
        idx = context_text.find(pat)
        if idx >= 0:
            # Grab surrounding context for the value
            snippet = context_text[max(0, idx-10):idx+80].strip()
            # Only count it if there's an actual number nearby, not just a label
            if re.search(r'[\d.]+', snippet):
                return snippet
    return None


def _context_preview(context_chunks: list[dict], max_chars: int = 6000) -> str:
    parts = []
    for c in context_chunks:
        if c.get("status") != "success":
            continue
        parts.append(f"[{c.get('tool')}] {c.get('content', '')}")
    joined = "\n\n".join(parts)
    return joined[:max_chars] + ("\n[...truncated]" if len(joined) > max_chars else "")


def _run_one_compute_step(step: dict, context_chunks: list[dict]) -> dict:
    tool_name = step.get("tool")
    step_num = step.get("step")
    purpose = step.get("purpose", "")
    planned_args = step.get("args", {})

    llm = get_llm(temperature=0.0, max_tokens=1024,
                   with_tools=[TOOLS_BY_NAME[tool_name]], tool_choice=tool_name)

    context = _context_preview(context_chunks)
    user_content = (
        f"METRIC/STEP PURPOSE: {purpose}\n"
        f"PLANNED ARGS (may be incomplete — fill in real numbers from context below): "
        f"{json.dumps(planned_args)}\n\n"
        f"RESEARCH CONTEXT GATHERED SO FAR:\n{context}\n\n"
        f"Call {tool_name} now with the correct numeric arguments."
    )
    messages = [SystemMessage(content=COMPUTE_SYSTEM_PROMPT), HumanMessage(content=user_content)]

    retries_used = 0
    result = None
    category = None
    for attempt in range(LOCAL_RETRY_LIMIT + 1):
        try:
            response = invoke_with_retry(llm, messages)
        except Exception as exc:
            # This is the actual crash point from the field report: Groq's
            # tool_choice forcing is not a hard guarantee — the underlying
            # model can still generate a call to a tool that isn't bound
            # (e.g. 'web_search' when only 'calculator' was in scope), and
            # Groq validates that server-side and raises a 400 instead of
            # returning a malformed response we could inspect client-side.
            # Previously this exception had no handler at this level and
            # took down the entire graph run. Now it's treated exactly
            # like any other failed attempt: classified, retried once
            # locally with a stronger instruction, and never escalated to
            # a replan (the tool itself isn't broken — the model just
            # deviated from the forced choice once).
            result = f"[Error] LLM call failed for forced tool '{tool_name}': {exc}"
            category = classify_result(tool_name, result)
            retries_used = attempt + 1
            if attempt < LOCAL_RETRY_LIMIT:
                messages.append(HumanMessage(
                    content=(
                        f"That request failed: {exc}\n"
                        f"You must call ONLY the tool named '{tool_name}' — "
                        f"do not reference or attempt to call any other tool. "
                        f"Try again with just a {tool_name} call."
                    )
                ))
            continue

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls or tool_calls[0]["name"] != tool_name:
            # Forced tool_choice should prevent this, but stay defensive.
            result = f"[Error] Model did not call {tool_name} as forced."
            category = "bad_args"
        else:
            call_args = tool_calls[0]["args"]
            fn = TOOLS_BY_NAME[tool_name].func
            try:
                result = fn(**call_args)
            except Exception as exc:
                result = f"[Error] {tool_name} raised an exception: {exc}"
            category = classify_result(tool_name, result)

        if not is_failure(category):
            break
        if category == SYSTEMIC:
            break

        # Before retrying: check if ALL required args are missing.
        # If so, the data simply isn't in the context — retrying with the
        # same context won't help. Skip immediately instead of wasting time.
        if category == "bad_args" and "Missing:" in str(result):
            missing_match = re.search(r"Missing:\s*([^.]+)", str(result))
            if missing_match:
                missing_args = [a.strip() for a in missing_match.group(1).split(",")]
                # Get the total required args for this metric
                try:
                    from tools.calculator import METRIC_REQUIRED_ARGS
                    metric = step.get("args", {}).get("metric", "")
                    required = METRIC_REQUIRED_ARGS.get(metric.lower(), [])
                    # If missing count >= 70% of required, data isn't there
                    if required and len(missing_args) >= max(1, int(0.7 * len(required))):
                        result = (f"[Calculator] Data unavailable — {len(missing_args)} of "
                                  f"{len(required)} required args ({', '.join(missing_args)}) "
                                  f"missing from fetched context. Skipping instead of retrying.")
                        category = "legitimate_empty"
                        print(f"  [ComputeExecutor] Most required args missing — skipping instead of retrying")
                        break
                except ImportError:
                    pass

        retries_used = attempt + 1
        # Give the model another shot, telling it what went wrong.
        messages.append(HumanMessage(
            content=f"That call failed: {result}\nTry again with corrected arguments."
        ))

    from utils.error_classifier import LEGITIMATE_EMPTY
    if category == LEGITIMATE_EMPTY:
        status = "skipped"
    elif not is_failure(category):
        status = "success"
    elif category == SYSTEMIC:
        status = "systemic_failure"
    else:
        status = "failed"

    return {
        "step": step_num, "tool": tool_name, "status": status,
        "content": result, "retries_used": retries_used,
        "error_category": category if is_failure(category) else None,
    }


def compute_executor_node(state: AgentState) -> dict:
    plan = state.get("roadmap", {}).get("research_plan", [])
    replan_count = state.get("replan_count", 0)

    existing_chunks = list(state.get("context_chunks", []))
    existing_calls = list(state.get("tool_calls_made", []))
    already_done_steps = {c["step"] for c in existing_chunks if c.get("status") == "success"}

    compute_steps = [
        s for s in plan
        if s.get("tool") in COMPUTE_TOOLS and s.get("step") not in already_done_steps
    ][:COMPUTE_STEP_CAP]

    needs_replan = False
    failed_step_for_replan = None

    # Build a single text view of all successful fetched context for the
    # "already available" check
    _all_context_text = "\n".join(
        str(c.get("content", "")) for c in existing_chunks
        if c.get("status") == "success"
    )

    for step in compute_steps:
        # Pre-check: does this metric already exist in the fetched data?
        metric = step.get("args", {}).get("metric", "")
        existing_snippet = _metric_already_available(metric, _all_context_text)
        if existing_snippet:
            print(f"[ComputeExecutor] step {step.get('step')} · {metric} SKIPPED — "
                  f"already in context: {existing_snippet[:100]}")
            entry = {
                "step": step.get("step"), "tool": "calculator",
                "status": "success",
                "content": f"[Calculator] Skipped — {metric} already available in fetched data: {existing_snippet}",
                "retries_used": 0, "error_category": None,
            }
            existing_chunks.append(entry)
            existing_calls.append({
                "step": entry["step"], "tool": entry["tool"],
                "status": entry["status"], "retries_used": 0,
            })
            continue

        entry = _run_one_compute_step(step, existing_chunks)
        existing_chunks.append(entry)
        existing_calls.append({
            "step": entry["step"], "tool": entry["tool"],
            "status": entry["status"], "retries_used": entry["retries_used"],
        })

        print(f"[ComputeExecutor] step {entry['step']} · {entry['tool']} · {entry['status']}"
              + (f" (retries: {entry['retries_used']})" if entry["retries_used"] else "")
              + (f"\n    → {str(entry['content'])[:200]}" if entry["status"] != "success" else ""))

        if entry["status"] == "systemic_failure" and replan_count < GLOBAL_REPLAN_BUDGET:
            needs_replan = True
            failed_step_for_replan = entry
            break

    return {
        "context_chunks": existing_chunks,
        "tool_calls_made": existing_calls,
        "needs_replan": needs_replan,
        "replan_reason": failed_step_for_replan,
        "replan_phase": "compute" if needs_replan else None,
    }
