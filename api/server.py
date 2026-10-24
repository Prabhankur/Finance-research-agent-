"""
api/server.py  –  FastAPI backend for the financial agent.

Two surfaces over one tool layer:

    POST /api/chat            conversational agent, streamed over SSE
    POST /api/report          the existing multi-page report pipeline, as a
                              background job the client polls

Why the report pipeline is a JOB and not a request:
It takes minutes and makes ~15 external API calls. Serving it from a request
handler means one dropped connection destroys the whole run with nothing to
show for it, and it pins a worker for the duration. Dispatching it to a
background thread with a polled status endpoint means the client can navigate
away, reconnect, and still collect the artifact.

Why SSE and not WebSockets for chat:
The chat stream is strictly one-directional server->client for the duration of
a turn (the user's input arrived in the POST body). SSE gives that with plain
HTTP: no upgrade handshake, no ping/pong keepalive logic, no reconnection
protocol to hand-roll, and it works through any proxy that speaks HTTP/1.1.
WebSockets would be the right call if the client needed to interrupt a running
turn — worth revisiting when cancel-mid-generation gets built.

Run:
    uvicorn api.server:app --reload --port 8000
    open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from config import GRAPH_RECURSION_LIMIT, REPORT_TYPES, REPORT_TYPE_LABELS
from database import thread_store
from graph.chat_graph import chat_agent_graph
from tools.chart_tool import CHART_TOOL_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Financial Agent API",
    description="Conversational financial research agent + report generation pipeline.",
    version="1.0.0",
)

# Wide-open CORS is fine for a local dev/demo tool. Lock this to specific
# origins before this is reachable from anywhere but localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: Optional[str] = Field(
        None, description="Omit to start a new conversation; reuse to continue one."
    )


class ReportRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    report_type: Optional[str] = Field(
        None, description=f"One of {REPORT_TYPES}. Omit to let the classifier decide."
    )
    thread_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse(event: str, data: Any) -> str:
    """Formats one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


# Human-readable labels for the "what is it doing right now" indicator.
_NODE_LABELS = {
    "intent_router": "Understanding your question",
    "chat_agent": "Thinking",
    "chat_tools": "Fetching data",
    "smalltalk": "Thinking",
    "report_handoff": "Preparing report",
}


async def _chat_event_stream(thread_id: str, message: str):
    """
    Runs one chat turn and yields SSE frames.

    The graph's nodes are synchronous and network-blocking, so the run happens
    on a worker thread and pushes into an asyncio.Queue rather than being
    awaited directly. Calling a blocking graph from the event loop would stall
    every other request on this process for the whole turn — including the
    report status polls, which is exactly when you least want the server to go
    unresponsive.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    inputs = {"messages": [HumanMessage(content=message)]}

    def _run() -> None:
        try:
            for update in chat_agent_graph.stream(inputs, config, stream_mode="updates"):
                loop.call_soon_threadsafe(queue.put_nowait, ("update", update))
        except Exception as exc:
            traceback.print_exc()
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("eof", None))

    threading.Thread(target=_run, daemon=True).start()

    yield _sse("start", {"thread_id": thread_id})

    # Title is set from the FIRST message of a NEW thread only — touch()
    # itself enforces that (see database/thread_store.py), so passing the
    # current message every call is safe and simpler than tracking "is this
    # thread new" here too.
    thread_store.touch(thread_id, first_message=message)

    final_answer: Optional[str] = None
    tool_events: list[dict] = []

    while True:
        kind, payload = await queue.get()

        if kind == "eof":
            break

        if kind == "error":
            yield _sse("error", {"message": payload})
            break

        # payload is {node_name: state_update}
        for node_name, node_update in (payload or {}).items():
            yield _sse(
                "status",
                {"node": node_name, "label": _NODE_LABELS.get(node_name, node_name)},
            )

            if not isinstance(node_update, dict):
                continue

            # Fires exactly once, when intent_router stages a NEW report
            # request (not on the later confirm turn — see intent_router_node,
            # only the REPORT branch sets pending_report_query to a real
            # value; CHAT/SMALLTALK explicitly null it, and CONFIRM_REPORT
            # doesn't touch the key at all). The web UI uses this to render
            # clickable report-type buttons instead of making the user type
            # the exact type name — clicking a button just sends that type's
            # label as a normal chat message, which _match_report_type_override
            # in nodes/intent_router.py resolves the same way free text would.
            if node_name == "intent_router" and node_update.get("pending_report_query"):
                yield _sse(
                    "report_offer",
                    {
                        "query": node_update["pending_report_query"],
                        "suggested_type": node_update.get("pending_report_type"),
                        "types": REPORT_TYPE_LABELS,
                    },
                )

            # Contradiction notes surface here so the UI can show a
            # "cross-source disagreement" banner alongside the answer. The
            # detector already ran deterministically in chat_agent_node and
            # prepended structured instructions to the synthesis prompt —
            # this event is purely for user-visible transparency, not
            # anything the model reads.
            for note in node_update.get("contradictions_detected", []) or []:
                yield _sse("contradiction", note)

            # Surface each tool call as it completes, so the user sees real
            # activity instead of a spinner during a 20-second fetch.
            for trace in node_update.get("tool_trace", []) or []:
                tool_events.append(trace)
                yield _sse("tool", trace)

                # render_chart is special: its `args` (labels/series/values)
                # is the actual payload the frontend needs to draw a chart,
                # not just a status chip. `args` is never trimmed the way
                # `content` is (see nodes/chat_agent.py's _run_one), so the
                # full structured spec survives intact all the way here —
                # this never touches the conversation's token budget at all.
                if trace.get("tool") == CHART_TOOL_NAME and trace.get("status") == "success":
                    yield _sse("chart_data", trace.get("args", {}))

            for msg in node_update.get("messages", []) or []:
                if isinstance(msg, AIMessage) and not (getattr(msg, "tool_calls", None) or []):
                    content = str(msg.content or "").strip()
                    if content:
                        final_answer = content
                elif isinstance(msg, AIMessage):
                    yield _sse(
                        "tool_calls",
                        {"calls": [tc["name"] for tc in msg.tool_calls]},
                    )

    if final_answer:
        # Chunked emission so text appears progressively. This is not
        # token-level streaming from the model — LangGraph's "messages" stream
        # mode requires the node to use .stream() rather than .invoke(), and
        # invoke is what makes the rate-limit retry wrapper work reliably.
        # Perceived latency is identical; swapping to true token streaming is
        # a contained change inside chat_agent_node.
        step = 24
        for i in range(0, len(final_answer), step):
            yield _sse("token", {"text": final_answer[i : i + step]})
            await asyncio.sleep(0.012)

    # Read the committed state to decide whether a report was confirmed.
    report_payload = None
    try:
        snapshot = chat_agent_graph.get_state(config)
        values = snapshot.values or {}
        if values.get("report_confirmed"):
            report_payload = {
                "query": values.get("pending_report_query") or message,
                "report_type": values.get("pending_report_type"),
            }
    except Exception as exc:
        print(f"[API] Could not read graph state: {exc}")

    if report_payload:
        # The client reacts to this by POSTing /api/report — the graph never
        # launches the pipeline itself.
        yield _sse("report_requested", report_payload)

    yield _sse(
        "done",
        {
            "thread_id": thread_id,
            "answer": final_answer or "",
            "tools_used": tool_events,
        },
    )

    # Post-turn evaluation. Fire-and-forget so the SSE stream can close
    # immediately — the judge call takes a couple of seconds, and holding the
    # HTTP connection open for it would just look like slowness on the user's
    # end for a signal they don't need in real time. Only score turns that
    # actually completed successfully; a synthesis failure isn't something
    # to grade, it's something to fix.
    try:
        from config import EVAL_ENABLED

        # Checked BEFORE reading graph state or importing the evaluator, so
        # when evaluation is off there is genuinely zero extra work — no
        # Groq call, no state read for scoring purposes. See config.py's
        # EVAL_ENABLED note for why this is a per-turn budget decision.
        if EVAL_ENABLED:
            snapshot = chat_agent_graph.get_state(config)
            values = snapshot.values or {}
            if final_answer and values.get("turn_status") == "COMPLETE":
                from evaluation.evaluator import score_turn_async

                score_turn_async(
                    turn_id=f"{thread_id}-{int(time.time())}",
                    thread_id=thread_id,
                    question=message,
                    answer=final_answer,
                    successful_tool_results=values.get("successful_tool_results") or [],
                    contradictions=values.get("contradictions_detected") or [],
                    turn_status=values.get("turn_status") or "",
                )
    except Exception as exc:
        print(f"[API] Post-turn scoring dispatch failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Chat endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Streams one conversational turn as Server-Sent Events."""
    thread_id = request.thread_id or f"thread-{uuid.uuid4().hex[:12]}"

    return StreamingResponse(
        _chat_event_stream(thread_id, request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx from buffering the stream into one blob at the end,
            # which silently defeats SSE behind a reverse proxy.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/threads")
async def list_threads(limit: int = 50):
    """
    Recently active conversations, most recent first — powers the sidebar.

    Reads from thread_store (see database/thread_store.py), NOT from the
    LangGraph checkpointer — the checkpointer has no efficient "list all
    threads" query, only point lookups by a known thread_id. Keeping a small
    separate metadata table is what makes this endpoint a single indexed
    query instead of a full checkpoint-store scan.
    """
    return {"threads": thread_store.list_threads(limit=limit)}


@app.get("/api/threads/{thread_id}/history")
async def history(thread_id: str):
    """Returns the stored transcript for a thread, for reloading the UI."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = chat_agent_graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"State read failed: {exc}")

    values = snapshot.values or {}
    messages = []
    for msg in values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage) and not (getattr(msg, "tool_calls", None) or []):
            if str(msg.content or "").strip():
                messages.append({"role": "assistant", "content": str(msg.content)})
        elif isinstance(msg, ToolMessage):
            messages.append(
                {"role": "tool", "name": msg.name, "chars": len(str(msg.content))}
            )

    return {
        "thread_id": thread_id,
        "messages": messages,
        "active_ticker": values.get("active_ticker"),
        "exists": bool(values),
    }


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """
    Removes a conversation from the sidebar.

    Deletes the thread_store metadata row (what makes it show up in the
    list) and best-effort clears LangGraph's own checkpoint state for that
    thread_id — SqliteSaver has no first-class "delete a thread" API in the
    version pinned here, so this reaches into its underlying connection
    directly. If that ever breaks across a langgraph-checkpoint-sqlite
    version bump, the practical impact is limited to orphaned checkpoint
    rows (harmless, just unreachable disk space) — the thread still
    disappears from the UI either way, which is what the user asked for.
    """
    removed = thread_store.delete_thread(thread_id)

    try:
        checkpointer = chat_agent_graph.checkpointer
        if checkpointer is not None and hasattr(checkpointer, "conn"):
            conn = checkpointer.conn
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.commit()
    except Exception as exc:
        print(f"[API] Could not clear checkpoint state for {thread_id}: {exc}")

    return {"thread_id": thread_id, "deleted": removed}


# ─────────────────────────────────────────────────────────────────────────────
#  Report jobs
# ─────────────────────────────────────────────────────────────────────────────

from database import report_jobs as _report_store


def _run_report_job(job_id: str, query: str, report_type: Optional[str]) -> None:
    """Executes the report pipeline on a worker thread and records the result."""
    # Imported lazily: the report graph pulls in matplotlib, python-docx and
    # edgartools at import time. Doing that at module load would add seconds to
    # server startup for a feature most requests never touch.
    from graph.financial_agent_graph import financial_agent

    _report_store.update(job_id, status="running", started_at=datetime.now().isoformat())
    started = time.time()

    initial_state = {
        "messages": [],
        "raw_query": query,
        "ticker": "",
        "elaborated_query": "",
        "report_type": report_type or "",
        "report_type_source": "user" if report_type else "",
        "roadmap": {},
        "tool_calls_made": [],
        "context_chunks": [],
        "iteration": 0,
        "disabled_tools": [],
        "replan_count": 0,
        "needs_replan": False,
        "replan_reason": None,
        "replan_phase": None,
        "rag_context": "",
        "report_outline": {},
        "final_report": "",
        "report_path": "",
    }

    try:
        result = financial_agent.invoke(
            initial_state, config={"recursion_limit": GRAPH_RECURSION_LIMIT}
        )
        # report_markdown is intentionally NOT persisted to the DB — it can
        # be tens of kilobytes per row and the file on disk (report_path) is
        # the authoritative artifact. Kept in-process only for the
        # /api/report/{job}/markdown endpoint's convenience.
        _REPORT_MARKDOWN[job_id] = result.get("final_report", "")
        _report_store.update(
            job_id,
            status="completed",
            finished_at=datetime.now().isoformat(),
            duration_s=round(time.time() - started, 1),
            report_path=result.get("report_path", "") or "",
            report_type=result.get("report_type", report_type),
        )
        print(f"[API] Report job {job_id} completed in {time.time() - started:.0f}s")
    except Exception as exc:
        traceback.print_exc()
        _report_store.update(
            job_id,
            status="failed",
            finished_at=datetime.now().isoformat(),
            duration_s=round(time.time() - started, 1),
            error=f"{type(exc).__name__}: {exc}",
        )


# In-process cache of report markdown text keyed by job_id — see _run_report_job
# for why this isn't in the DB. Lost on restart, which is fine: markdown is a
# derived view of the file at report_path, which IS persisted.
_REPORT_MARKDOWN: dict[str, str] = {}


@app.post("/api/report")
async def create_report(request: ReportRequest):
    """Dispatches a report run and returns a job id to poll."""
    if request.report_type and request.report_type not in REPORT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"report_type must be one of {REPORT_TYPES}",
        )

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    _report_store.create(
        job_id=job_id,
        query=request.query,
        report_type=request.report_type,
        thread_id=request.thread_id,
    )

    threading.Thread(
        target=_run_report_job,
        args=(job_id, request.query, request.report_type),
        daemon=True,
    ).start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/report/{job_id}")
async def report_status(job_id: str):
    """Polling endpoint. Omits the full markdown to keep polls cheap."""
    job = _report_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")

    payload = dict(job)
    payload["has_markdown"] = job_id in _REPORT_MARKDOWN
    if job.get("report_path"):
        payload["download_url"] = f"/api/report/{job_id}/download"
    return payload


@app.get("/api/report/{job_id}/markdown")
async def report_markdown(job_id: str):
    job = _report_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job.get('status')}")
    # If the process restarted since the job completed, in-memory markdown
    # is gone. Point the caller at the file — that's still there.
    if job_id not in _REPORT_MARKDOWN:
        raise HTTPException(
            status_code=410,
            detail="Markdown not in memory (server restarted). Use /download for the .docx.",
        )
    return {"job_id": job_id, "markdown": _REPORT_MARKDOWN[job_id]}


@app.get("/api/report/{job_id}/download")
async def report_download(job_id: str):
    job = _report_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")

    path_str = job.get("report_path") or ""
    if not path_str:
        raise HTTPException(status_code=409, detail="Report not ready")

    path = Path(path_str)
    # The saver returns a .md path; prefer the .docx sibling when it exists,
    # since that's what a reader actually wants to open.
    docx = path.with_suffix(".docx")
    target = docx if docx.exists() else path

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File missing: {target}")

    return FileResponse(path=str(target), filename=target.name)


@app.get("/api/reports")
async def list_reports():
    """All recent reports, most recent first. Persists across restart —
    was in-memory before."""
    return {"jobs": _report_store.list_all(limit=50)}


@app.get("/api/threads/{thread_id}/reports")
async def list_reports_for_thread(thread_id: str):
    """Reports generated FROM a specific conversation — powers the "reports
    from this chat" panel next to the sidebar, closing the gap where
    reports had no visible link back to the thread that requested them."""
    return {"thread_id": thread_id, "jobs": _report_store.list_for_thread(thread_id)}


@app.get("/api/evaluation/rollup")
async def evaluation_rollup(days: int = 7):
    """Aggregate scoring report across recent turns — the 'what should we
    improve' view."""
    from evaluation.evaluator import rollup_report
    return rollup_report(days=days)


@app.get("/api/evaluation/recent")
async def evaluation_recent(limit: int = 20):
    """Recent per-turn scores — for the debug UI panel."""
    from evaluation.evaluator import recent_scores
    return {"scores": recent_scores(limit=limit)}


@app.get("/api/threads/{thread_id}/evaluation")
async def evaluation_for_thread(thread_id: str):
    """All scored turns from ONE conversation, chronological — answers
    'how did quality trend across this specific chat'. The thread_id
    column has existed since the evaluator was built; this is the query
    endpoint for it that was missing."""
    from evaluation.evaluator import scores_for_thread
    return {"thread_id": thread_id, "scores": scores_for_thread(thread_id)}


# ─────────────────────────────────────────────────────────────────────────────
#  Meta
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Reports which data sources are actually usable, not just liveness."""
    keys = {
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "alpha_vantage": bool(os.getenv("ALPHA_VANTAGE_KEY")),
        "tavily": bool(os.getenv("TAVILY_API_KEY")),
        "fred": bool(os.getenv("FRED_API_KEY")),
        "sec_user_agent": bool(os.getenv("SEC_USER_AGENT")),
    }
    running = sum(1 for j in _report_store.list_all(limit=50) if j.get("status") == "running")

    # Cache stats prove the knowledge layer is actually being hit rather
    # than silently missing on every call — the difference is invisible from
    # the outside otherwise, and a cache that never hits looks identical to
    # no cache at all.
    try:
        from database.cache_service import cache_stats
        cache = cache_stats()
    except Exception as exc:
        cache = {"error": str(exc)}

    # Live model catalog: the authoritative answer to "which models can this
    # key actually use", which is what a string of 404s came down to.
    try:
        from utils.groq_client import list_available_models
        models = list_available_models()
    except Exception:
        models = []

    try:
        from config import EVAL_ENABLED
        eval_enabled = EVAL_ENABLED
    except Exception:
        eval_enabled = None

    return {
        "status": "ok" if keys["groq"] else "degraded",
        "keys_present": keys,
        "report_types": REPORT_TYPE_LABELS,
        "running_report_jobs": running,
        "cache": cache,
        "available_models": models[:20],
        "eval_enabled": eval_enabled,
    }


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
