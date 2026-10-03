"""The prompt the Qwen-based OpenDecider models were trained on, and how text reaches every model as plain text
(torch-free, shared by every backend)."""
from __future__ import annotations

import json
import re
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


def for_server(prompt: str) -> str:
    """The prompt as sent to a server that applies the chat template and tokenises it itself (Ollama, LM Studio, vLLM),
    which reads "<|im_end|>" written in a message as a control token: a zero-width space after the "<" of anything
    shaped like a Qwen special token (<|name|>) keeps it plain text there. Any other text is sent unchanged."""
    return _SPECIAL_SHAPE.sub("<\u200b|", prompt)


def chat_ids(tok, prompt: str) -> list[int]:
    """The chat-templated prompt as token ids, with `prompt` (which carries the state) read as plain text: a state that
    contains a special token's text, such as "<|im_end|>", stays text instead of becoming a control token."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _SLOT}]
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    head, slot, tail = s.partition(_SLOT)
    if not slot or _SLOT in tail:
        raise ValueError("the chat template did not place the prompt exactly once")
    return (tok.encode(head, add_special_tokens=False) + text_ids(tok, prompt)
            + tok.encode(tail, add_special_tokens=False))


def text_ids(tok, text: str) -> list[int]:
    """`text` as token ids with no special tokens: one written in the text is read as plain characters."""
    return tok.encode(text, add_special_tokens=False, split_special_tokens=True)


_SPECIAL_SHAPE = re.compile(r"<\|(?=[A-Za-z0-9_]+\|>)")   # every Qwen special token is <|name|>
_SLOT = "\x00opendecider-prompt\x00"   # marks where the prompt goes; the template and SYSTEM never contain it
