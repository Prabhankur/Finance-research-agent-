"""evaluation/  –  LLM-as-judge scoring for real conversation turns.

See evaluator.py for the rubric, scoring, and rollup. This package is
deliberately separate from database/ — the evaluator writes to the same DB
but is conceptually a quality-signal layer, not part of the knowledge
layer's read/write path.
"""
