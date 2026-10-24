"""
nodes/__init__.py – Exports every node function the graph wires together.

This used to also contain a second, dead copy of the entire tool registry
(ALL_TOOLS / TOOL_NODE / TOOL_DESCRIPTIONS, duplicating tools/__init__.py)
built from duplicate tool implementation files that lived in this same
folder. Neither was ever imported by the graph — graph/financial_agent_graph.py
always imported ALL_TOOLS from `tools`, not `nodes`. Both the dead registry
and the duplicate tool files have been removed; this file now only exports
node functions.
"""

from .query_elaboration import query_elaboration_node
from .report_type_classifier import report_type_classifier_node
from .roadmap_planner    import roadmap_planner_node
from .fetch_executor     import fetch_executor_node
from .compute_executor   import compute_executor_node
from .replanner           import replanner_node
from .context_router     import context_router_node
from .report_generator   import report_generator_node
from .report_saver       import report_saver_node

__all__ = [
    "query_elaboration_node",
    "report_type_classifier_node",
    "roadmap_planner_node",
    "fetch_executor_node",
    "compute_executor_node",
    "replanner_node",
    "context_router_node",
    "report_generator_node",
    "report_saver_node",
]
