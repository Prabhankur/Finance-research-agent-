"""
chat_cli.py  –  Terminal client for the chat agent.

Talks to the compiled graph directly (no HTTP), which makes it the fastest way
to test agent behaviour: a bad prompt or a broken tool shows up here in one
command, with the node print()s inline, instead of being filtered through SSE
frames and a browser console.

Usage:
    python chat_cli.py
    python chat_cli.py --ask "What is Apple's free cash flow?"
    python chat_cli.py --thread my-session        # resume a conversation
    python chat_cli.py --trace                    # show tool timings per turn

Commands inside the REPL:
    /new      start a fresh thread
    /trace    toggle the tool-timing panel
    /state    dump routing state for the current thread
    /quit     exit
"""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage

from config import GRAPH_RECURSION_LIMIT


# ANSI colours, disabled automatically when piping to a file.
_TTY = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


DIM = lambda s: c(s, "2")
BOLD = lambda s: c(s, "1")
BLUE = lambda s: c(s, "34")
GREEN = lambda s: c(s, "32")
RED = lambda s: c(s, "31")
YELLOW = lambda s: c(s, "33")


def run_turn(graph, thread_id: str, message: str, show_trace: bool) -> None:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    answer = ""
    traces: list[dict] = []

    for update in graph.stream(
        {"messages": [HumanMessage(content=message)]}, config, stream_mode="updates"
    ):
        for node_name, node_update in (update or {}).items():
            if not isinstance(node_update, dict):
                continue

            traces.extend(node_update.get("tool_trace", []) or [])

            for msg in node_update.get("messages", []) or []:
                if isinstance(msg, AIMessage):
                    calls = getattr(msg, "tool_calls", None) or []
                    if calls:
                        names = ", ".join(tc["name"] for tc in calls)
                        print(DIM(f"   ↳ calling: {names}"))
                    elif str(msg.content or "").strip():
                        answer = str(msg.content).strip()

    print()
    print(BOLD(GREEN("Agent:")), answer or DIM("(no answer produced)"))

    if show_trace and traces:
        print()
        print(DIM("   tools used:"))
        for t in traces:
            colour = {"success": GREEN, "empty": YELLOW}.get(t["status"], RED)
            print(
                DIM("   • ")
                + colour(f"{t['tool']:<20}")
                + DIM(f"{t['status']:<9} {t['duration_s']:>5}s  {t['chars']:>7} chars")
                + (DIM("  [truncated]") if t.get("truncated") else "")
            )

    # Surface the report handoff so the CLI isn't silently dropping the flow.
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    if values.get("report_confirmed"):
        print()
        print(YELLOW("   ⚑ Report confirmed. Run the pipeline with:"))
        print(
            DIM(f'     python main.py --query "{values.get("pending_report_query")}"')
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial chat agent (terminal)")
    parser.add_argument("--ask", help="Ask one question and exit")
    parser.add_argument("--thread", help="Thread id to resume")
    parser.add_argument("--trace", action="store_true", help="Show tool timings")
    args = parser.parse_args()

    from graph.chat_graph import chat_agent_graph

    thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"
    show_trace = args.trace

    if args.ask:
        run_turn(chat_agent_graph, thread_id, args.ask, show_trace)
        return

    print()
    print(BOLD("  Financial Research Agent") + DIM("  ·  chat mode"))
    print(DIM(f"  thread: {thread_id}"))
    print(DIM("  /new  /trace  /state  /quit"))
    print()

    while True:
        try:
            message = input(BOLD(BLUE("You: "))).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + DIM("bye"))
            return

        if not message:
            continue

        if message in ("/quit", "/exit", "/q"):
            print(DIM("bye"))
            return
        if message == "/new":
            thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            print(DIM(f"  new thread: {thread_id}\n"))
            continue
        if message == "/trace":
            show_trace = not show_trace
            print(DIM(f"  trace {'on' if show_trace else 'off'}\n"))
            continue
        if message == "/state":
            snapshot = chat_agent_graph.get_state(
                {"configurable": {"thread_id": thread_id}}
            )
            values = snapshot.values or {}
            print(
                DIM(
                    f"  intent={values.get('intent')} "
                    f"source={values.get('intent_source')} "
                    f"ticker={values.get('active_ticker')} "
                    f"messages={len(values.get('messages', []))} "
                    f"pending_report={values.get('pending_report_type')}\n"
                )
            )
            continue

        try:
            run_turn(chat_agent_graph, thread_id, message, show_trace)
        except Exception as exc:
            print(RED(f"  turn failed: {type(exc).__name__}: {exc}"))
        print()


if __name__ == "__main__":
    main()
