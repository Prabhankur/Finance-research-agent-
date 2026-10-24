"""
ui/app.py  –  Streamlit dashboard for the Autonomous Financial Report Agent.

Run from the financial_agent/ root:
    streamlit run ui/app.py

Features:
    • Query input with example queries
    • Live node-by-node progress log
    • Report preview (rendered markdown)
    • Download report as .md and .docx
    • Run history (session-based)
    • LangSmith trace link per run
"""

from __future__ import annotations
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# ── Make sure project root is on sys.path ─────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Financial Report Agent",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1d2e;
    border-right: 1px solid #2d3748;
}

/* Cards */
.metric-card {
    background: #1e2130;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* Node log entries */
.log-entry {
    background: #1a1d2e;
    border-left: 3px solid #4ade80;
    border-radius: 4px;
    padding: 8px 14px;
    margin: 6px 0;
    font-family: monospace;
    font-size: 13px;
    color: #e2e8f0;
}
.log-entry.running {
    border-left-color: #facc15;
    animation: pulse 1s infinite;
}
.log-entry.error {
    border-left-color: #f87171;
}

/* Report preview */
.report-preview {
    background: #1e2130;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 24px;
    max-height: 600px;
    overflow-y: auto;
}

/* History items */
.history-item {
    background: #1e2130;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    cursor: pointer;
}
.history-item:hover { border-color: #4ade80; }

/* Status badges */
.badge-success { color: #4ade80; font-weight: bold; }
.badge-running { color: #facc15; font-weight: bold; }
.badge-error   { color: #f87171; font-weight: bold; }

/* Headings */
h1, h2, h3 { color: #f1f5f9 !important; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_session():
    defaults = {
        "history":       [],   # list of completed run dicts
        "current_run":   None, # dict for in-progress run
        "selected_run":  None, # run shown in preview panel
        "is_running":    False,
        "log_lines":     [],
        "langsmith_url": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Financial Agent")
    st.markdown("---")

    # ── LangSmith status ──────────────────────────────────────────────────────
    ls_key     = os.getenv("LANGCHAIN_API_KEY", "")
    ls_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    ls_project = os.getenv("LANGCHAIN_PROJECT", "financial-agent")

    if ls_key and ls_tracing:
        st.markdown(
            f'<span class="badge-success">● LangSmith ON</span>  '
            f'<br><small>Project: <code>{ls_project}</code></small>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"[Open LangSmith ↗](https://smith.langchain.com/projects/{ls_project})",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge-error">● LangSmith OFF</span>'
            '<br><small>Add LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true to .env</small>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── API key status ────────────────────────────────────────────────────────
    st.markdown("**API Keys**")
    key_checks = {
        "GROQ_API_KEY":      "Groq LLM",
        "ALPHA_VANTAGE_KEY": "Alpha Vantage",
        "TAVILY_API_KEY":    "Tavily Search",
        "FRED_API_KEY":      "FRED Macro Data",
    }
    for env_var, label in key_checks.items():
        val = os.getenv(env_var, "")
        status = "✅" if val else "❌"
        st.markdown(f"{status} {label}")
    st.caption(
        "Only Groq is a hard requirement. Missing Alpha Vantage, Tavily, or "
        "FRED keys just disable the tool that needs them for this run — the "
        "agent routes around it and notes the gap in the report instead of failing."
    )

    st.markdown("---")

    # ── Report type dropdown ──────────────────────────────────────────────────
    st.markdown("**Report Type** (optional)")
    _REPORT_TYPE_OPTIONS = {
        "Auto (let the agent decide)": "",
        "Company / Stock Research":    "COMPANY_RESEARCH",
        "Comparative Analysis (2+ companies)": "COMPARATIVE_ANALYSIS",
        "Earnings / Event Analysis":   "EARNINGS_EVENT",
        "Valuation / Risk Deep-Dive":  "VALUATION_RISK",
        "Market / Macro / Country":    "MACRO_COUNTRY",
        "Other / Custom":              "OTHER",
    }
    _picked_label = st.selectbox(
        "Pre-select a report archetype",
        options=list(_REPORT_TYPE_OPTIONS.keys()),
        index=0,
        help=(
            "Auto lets an LLM classifier pick the archetype from your query. "
            "Pre-selecting overrides that and skips the classification step."
        ),
    )
    st.session_state["_selected_report_type"] = _REPORT_TYPE_OPTIONS[_picked_label]

    st.markdown("---")

    # ── Example queries ───────────────────────────────────────────────────────
    st.markdown("**Example Queries**")
    examples = [
        "Analyse Apple AAPL for FY2024",
        "Analyse Microsoft MSFT for FY2024",
        "Analyse Tesla TSLA for FY2024",
        "Analyse NVIDIA NVDA for FY2024",
        "Analyse Amazon AMZN for FY2024",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill_query"] = ex
            st.rerun()

    st.markdown("---")

    # ── History ───────────────────────────────────────────────────────────────
    st.markdown("**Run History**")
    if not st.session_state["history"]:
        st.caption("No runs yet.")
    else:
        for i, run in enumerate(reversed(st.session_state["history"])):
            label = f"📄 {run['ticker']} — {run['timestamp']}"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state["selected_run"] = run
                st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 📊 Autonomous Financial Report Agent")
st.markdown(
    "Powered by **LangGraph** · **Groq (Llama 3.3 70B)** · **LangSmith**"
)
st.markdown("---")

# ── Query input row ───────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
col_input, col_btn = st.columns([5, 1])

with col_input:
    query = st.text_input(
        label       = "Research Query",
        value       = prefill,
        placeholder = "e.g. Analyse Apple AAPL for FY2024",
        label_visibility = "collapsed",
    )

with col_btn:
    run_clicked = st.button(
        "🚀 Run",
        type             = "primary",
        use_container_width = True,
        disabled         = st.session_state["is_running"],
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_live, tab_report, tab_history = st.tabs([
    "⚡ Live Progress",
    "📄 Report Preview",
    "🕘 History",
])


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: stream the agent and yield per-node updates
# ─────────────────────────────────────────────────────────────────────────────

NODE_LABELS = {
    "query_elaboration":       ("🔍", "Elaborating query"),
    "report_type_classifier":  ("🏷️",  "Classifying report type"),
    "roadmap_planner":         ("🗺️",  "Planning research roadmap"),
    "fetch_executor":          ("📡", "Fetching data (direct, no LLM per step)"),
    "compute_executor":        ("🧮", "Computing derived metrics"),
    "replanner":               ("🛠️",  "Recovering from a failed data source"),
    "context_router":          ("🗂️",  "Routing context to report sections"),
    "report_generator":        ("✍️",  "Generating report sections"),
    "report_saver":            ("💾", "Saving report files"),
}

def _node_label(node_name: str) -> tuple[str, str]:
    return NODE_LABELS.get(node_name, ("⚙️", node_name))


def _run_agent_streaming(query: str, report_type: str = ""):
    """
    Generator that runs the agent in streaming mode and yields
    progress dicts after each node completes.
    Each dict: {node, icon, label, detail, elapsed, state}

    Args:
        query: user's research question
        report_type: optional pre-selected report archetype (skips classifier LLM call)
    """
    # Boot LangSmith
    from main import _setup_langsmith
    _setup_langsmith()

    from graph.financial_agent_graph import financial_agent

    initial_state = {
        "raw_query":          query,
        "messages":           [],
        "ticker":             "",
        "elaborated_query":   "",
        "report_type":        report_type,
        "report_type_source": "",
        "roadmap":            {},
        "tool_calls_made":    [],
        "context_chunks":     [],
        "iteration":          0,
        "disabled_tools":     [],
        "replan_count":       0,
        "needs_replan":       False,
        "replan_reason":      None,
        "replan_phase":       None,
        "rag_context":        "",
        "report_outline":     {},
        "final_report":       "",
        "report_path":        "",
    }

    t_start      = time.time()
    merged_state = dict(initial_state)

    try:
      for chunk in financial_agent.stream(initial_state, config={"recursion_limit": 100}):
        for node_name, node_output in chunk.items():
            merged_state.update(node_output)
            elapsed      = round(time.time() - t_start, 1)
            icon, label  = _node_label(node_name)

            # Build a short detail string
            detail = ""
            if node_name == "query_elaboration":
                detail = f"Ticker: {node_output.get('ticker', '?')}"
            elif node_name == "report_type_classifier":
                rt = node_output.get("report_type", "?")
                src = node_output.get("report_type_source", "?")
                detail = f"{rt} (source: {src})"
            elif node_name == "roadmap_planner":
                plan = node_output.get("roadmap", {}).get("research_plan", [])
                detail = f"{len(plan)} research steps planned"
            elif node_name == "fetch_executor":
                calls = node_output.get("tool_calls_made", [])
                ok = sum(1 for c in calls if c.get("status") == "success")
                failed = sum(1 for c in calls if c.get("status") in ("failed", "systemic_failure"))
                detail = f"{ok} succeeded, {failed} failed" if calls else "running…"
                if node_output.get("needs_replan"):
                    detail += " — escalating to replanner"
            elif node_name == "compute_executor":
                calls = node_output.get("tool_calls_made", [])
                if calls:
                    last = calls[-1]
                    detail = f"{last.get('tool', '?')}: {last.get('status', '?')}"
                if node_output.get("needs_replan"):
                    detail += " — escalating to replanner"
            elif node_name == "replanner":
                n = node_output.get("replan_count", "?")
                new_len = len(node_output.get("roadmap", {}).get("research_plan", []))
                detail = f"Replan #{n} — {new_len} steps in revised plan"
            elif node_name == "context_router":
                chars = len(merged_state.get("rag_context", ""))
                detail = f"~{chars:,} chars routed to sections"
            elif node_name == "report_generator":
                chars = len(node_output.get("final_report", ""))
                detail = f"~{chars:,} chars generated" if chars else "writing..."
            elif node_name == "report_saver":
                detail = node_output.get("report_path", "")

            yield {
                "node":    node_name,
                "icon":    icon,
                "label":   label,
                "detail":  detail,
                "elapsed": elapsed,
                "state":   merged_state,
            }

    except GeneratorExit:
        # Streamlit kills the generator when the page reruns (user clicked
        # something or the script re-executed). Not a bug, just the streaming
        # model stopping early. Yield whatever we have so the UI shows the
        # partial result instead of crashing.
        pass

    yield {"node": "__done__", "state": merged_state}


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 1 – Live Progress
# ─────────────────────────────────────────────────────────────────────────────
with tab_live:

    # ── Metrics row ───────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    run = st.session_state.get("selected_run") or st.session_state.get("current_run") or {}
    m1.metric("Ticker",       run.get("ticker", "—"))
    m2.metric("Tool Calls",   run.get("tool_calls", "—"))
    m3.metric("Report Size",  run.get("report_size", "—"))
    m4.metric("Duration",     run.get("duration", "—"))

    st.markdown("---")

    # ── Log container ─────────────────────────────────────────────────────────
    log_placeholder    = st.empty()
    status_placeholder = st.empty()

    def _render_logs(lines: list[dict]):
        html = ""
        for entry in lines:
            cls  = "log-entry"
            html += (
                f'<div class="{cls}">'
                f'{entry["icon"]} &nbsp; <b>{entry["label"]}</b>'
                f'{"&nbsp;–&nbsp;" + entry["detail"] if entry["detail"] else ""}'
                f'&nbsp;&nbsp;<small style="color:#94a3b8">[{entry["elapsed"]}s]</small>'
                f'</div>'
            )
        log_placeholder.markdown(html, unsafe_allow_html=True)


    # ── Run on button click ───────────────────────────────────────────────────
    if run_clicked and query.strip():
        if not os.getenv("GROQ_API_KEY"):
            st.error("❌ GROQ_API_KEY missing in .env — cannot run agent.")
        else:
            st.session_state["is_running"]  = True
            st.session_state["log_lines"]   = []
            st.session_state["current_run"] = {}
            st.session_state["selected_run"] = None

            t_start   = time.time()
            log_lines = []

            status_placeholder.info("⚡ Agent running…  (this takes 2-5 minutes)")

            try:
                _rt = st.session_state.get("_selected_report_type", "")
                for update in _run_agent_streaming(query, report_type=_rt):
                    if update["node"] == "__done__":
                        break

                    log_lines.append({
                        "icon":    update["icon"],
                        "label":   update["label"],
                        "detail":  update["detail"],
                        "elapsed": update["elapsed"],
                    })
                    _render_logs(log_lines)

                    # Cache final state along the way
                    st.session_state["current_run"] = update["state"]

                # ── Run finished ──────────────────────────────────────────────
                final = st.session_state.get("current_run", {})
                duration = round(time.time() - t_start, 1)

                completed_run = {
                    "query":       query,
                    "ticker":      final.get("ticker", "?"),
                    "timestamp":   datetime.now().strftime("%H:%M:%S"),
                    "tool_calls":  str(len(final.get("tool_calls_made", []))),
                    "report_size": f"~{len(final.get('final_report',''))//1000}K chars",
                    "duration":    f"{duration}s",
                    "report_path": final.get("report_path", ""),
                    "final_report": final.get("final_report", ""),
                    "log_lines":   log_lines,
                    "disabled_tools": final.get("disabled_tools", []),
                    "data_limitation_notes": final.get("roadmap", {}).get("data_limitation_notes", []),
                }

                st.session_state["history"].append(completed_run)
                st.session_state["selected_run"] = completed_run
                st.session_state["is_running"]   = False

                status_placeholder.success(
                    f"✅ Report complete for **{completed_run['ticker']}** "
                    f"in {duration}s — switch to **Report Preview** tab to read it."
                )
                if completed_run["disabled_tools"] or completed_run["data_limitation_notes"]:
                    gap_msg = "⚠️ This run completed with data gaps: "
                    if completed_run["disabled_tools"]:
                        gap_msg += f"tools unavailable this run: {', '.join(completed_run['disabled_tools'])}. "
                    for note in completed_run["data_limitation_notes"]:
                        gap_msg += f"\n- {note}"
                    st.warning(gap_msg)

                # Refresh metrics
                m1.metric("Ticker",      completed_run["ticker"])
                m2.metric("Tool Calls",  completed_run["tool_calls"])
                m3.metric("Report Size", completed_run["report_size"])
                m4.metric("Duration",    completed_run["duration"])

            except Exception as exc:
                st.session_state["is_running"] = False
                status_placeholder.error(f"❌ Agent error: {exc}")
                st.exception(exc)

    elif run_clicked and not query.strip():
        st.warning("Please enter a query first.")

    # Show last log if revisiting tab
    elif st.session_state.get("log_lines"):
        _render_logs(st.session_state["log_lines"])


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 2 – Report Preview
# ─────────────────────────────────────────────────────────────────────────────
with tab_report:
    sel = st.session_state.get("selected_run")

    if not sel:
        st.info("🔍 Run an analysis first — the report will appear here.")
    else:
        # ── Header row ────────────────────────────────────────────────────────
        hcol1, hcol2, hcol3 = st.columns([4, 1, 1])
        with hcol1:
            st.markdown(f"### 📄 Report: {sel.get('ticker', '?')} — {sel.get('timestamp', '')}")

        report_text = sel.get("final_report", "")
        report_path = sel.get("report_path", "")

        if sel.get("disabled_tools") or sel.get("data_limitation_notes"):
            gap_lines = []
            if sel.get("disabled_tools"):
                gap_lines.append(f"Tools unavailable this run: {', '.join(sel['disabled_tools'])}")
            gap_lines.extend(sel.get("data_limitation_notes", []))
            st.warning("⚠️ This report has data gaps:\n" + "\n".join(f"- {g}" for g in gap_lines))

        # ── Download buttons ──────────────────────────────────────────────────
        with hcol2:
            if report_text:
                st.download_button(
                    label     = "⬇️ Download .md",
                    data      = report_text.encode("utf-8"),
                    file_name = f"{sel.get('ticker','report')}_report.md",
                    mime      = "text/markdown",
                    use_container_width = True,
                )

        with hcol3:
            # Try to read the .docx bytes if it was saved
            if report_path:
                docx_path = report_path.replace(".md", ".docx")
                if os.path.exists(docx_path):
                    with open(docx_path, "rb") as f:
                        docx_bytes = f.read()
                    st.download_button(
                        label     = "⬇️ Download .docx",
                        data      = docx_bytes,
                        file_name = f"{sel.get('ticker','report')}_report.docx",
                        mime      = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width = True,
                    )

        st.markdown("---")

        # ── Rendered report ───────────────────────────────────────────────────
        if report_text:
            st.markdown(report_text, unsafe_allow_html=False)
        else:
            st.warning("Report text not available.")


# ─────────────────────────────────────────────────────────────────────────────
#  Tab 3 – History
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    history = st.session_state.get("history", [])

    if not history:
        st.info("No runs yet. Run an analysis to see history here.")
    else:
        st.markdown(f"**{len(history)} run(s) this session**")
        st.markdown("---")

        for i, run in enumerate(reversed(history)):
            with st.expander(
                f"📄  {run['ticker']}  ·  {run['query'][:60]}  ·  {run['timestamp']}",
                expanded = (i == 0),
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ticker",      run["ticker"])
                c2.metric("Tool Calls",  run["tool_calls"])
                c3.metric("Report Size", run["report_size"])
                c4.metric("Duration",    run["duration"])

                if st.button(
                    "📄 View this report",
                    key  = f"view_hist_{i}",
                    type = "secondary",
                ):
                    st.session_state["selected_run"] = run
                    st.rerun()

                # Mini log replay
                log_lines = run.get("log_lines", [])
                if log_lines:
                    st.markdown("**Execution log:**")
                    for entry in log_lines:
                        st.markdown(
                            f"`{entry['elapsed']}s` &nbsp; {entry['icon']} "
                            f"**{entry['label']}** — {entry['detail']}",
                            unsafe_allow_html=True,
                        )