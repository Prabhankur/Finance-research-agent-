"""
utils/text_budget.py — share a character budget fairly across several texts.

Used wherever several tool results must be squeezed into one prompt (the
rescue synthesis and the evaluation judge). Both used to divide "whatever is
left" by the TOTAL count on every pass, so the first item kept most of its text
and the last was cut hardest — with five companies, the fifth kept about 40% of
what the first did, and nothing said so.
"""
from __future__ import annotations


def fair_share(blocks: list[str], max_chars: int, floor: int = 200,
               marker: str = "\n…[truncated]") -> list[str]:
    """
    Returns `blocks` cut so their total fits `max_chars`, sharing the budget
    evenly: a block shorter than its fair share keeps everything and the budget
    it does not need flows to the longer ones. Anything cut ends with `marker`
    so the reader knows the text is incomplete. `floor` stops a block being
    squeezed to nothing when there are very many.
    """
    if sum(len(b) for b in blocks) <= max_chars:
        return list(blocks)
    n = len(blocks)
    allowance = [0] * n
    remaining = max_chars
    for rank, i in enumerate(sorted(range(n), key=lambda k: len(blocks[k]))):
        share = remaining // (n - rank)
        allowance[i] = min(len(blocks[i]), max(share, floor))
        remaining -= allowance[i]
    return [b if len(b) <= a else b[:a] + marker for b, a in zip(blocks, allowance)]
