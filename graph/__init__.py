"""
graph/__init__.py

State classes are imported eagerly — they're just TypedDicts, effectively free.

The compiled graphs are exposed LAZILY through module __getattr__. Importing
financial_agent_graph at package level used to mean that `from graph.chat_state
import ChatState` transitively imported the whole report pipeline: every node,
every tool, and therefore yfinance, matplotlib, python-docx and edgartools.

That had two real costs. Server startup paid several seconds of import time for
a pipeline most requests never touch. And a missing optional dependency in one
report-only tool would break the chat agent at import time, even though chat
never calls that tool — a failure with no logical connection to its cause,
which is the worst kind to debug.

`from graph import financial_agent` still works exactly as before; the import
just happens on first attribute access instead of at package load.
"""

from .state import AgentState
from .chat_state import ChatState

__all__ = [
    "AgentState",
    "ChatState",
    "financial_agent",
    "build_financial_agent_graph",
    "chat_agent_graph",
    "build_chat_graph",
]

_LAZY = {
    "financial_agent":            ("graph.financial_agent_graph", "financial_agent"),
    "build_financial_agent_graph": ("graph.financial_agent_graph", "build_financial_agent_graph"),
    "chat_agent_graph":           ("graph.chat_graph", "chat_agent_graph"),
    "build_chat_graph":           ("graph.chat_graph", "build_chat_graph"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_path, attribute = _LAZY[name]
        return getattr(importlib.import_module(module_path), attribute)
    raise AttributeError(f"module 'graph' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
