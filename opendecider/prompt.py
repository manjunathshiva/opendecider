"""The prompt the Qwen-based OpenDecider models were trained on (torch-free, shared by every backend)."""
from __future__ import annotations

import json
import string

LETTERS = string.ascii_uppercase
SYSTEM = "You make one decision for a software system."


def render(state, instructions, options: dict, lettered=True) -> str:
    st = state if isinstance(state, str) else json.dumps(state, indent=1, ensure_ascii=False)
    lines = []
    for i, (k, v) in enumerate(options.items()):
        d = f"{k}: {v}" if v and v != k else k
        lines.append(f"{LETTERS[i]}) {d}" if lettered else f"- {d}")
    ask = ("Answer with the letter of the correct option only." if lettered
           else "Answer with the exact option name only.")
    return f"Input:\n{st}\n\nQuestion: {instructions}\n\nOptions:\n" + "\n".join(lines) + f"\n\n{ask}"
