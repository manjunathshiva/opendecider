"""opendecider-nano's tokenizer for tokenizers.js (@opendecider/web), so it gives exactly the ids the Python package
gives (`opendecider.prompt.text_ids`):

1. The GPT-2 split as an explicit Split pre-tokenizer: tokenizers.js's built-in ByteLevel pattern uses JavaScript's \\s,
   which also matches U+FEFF, and the Rust tokenizer's does not.
2. No special added tokens: text with "[MASK]" in it stays text (in Python, split_special_tokens=True). The
   [CLS]/[SEP]/[MASK]/[PAD] ids are added by the prompt builder, never by the tokenizer.

    python packaging/onnx/make_web_tokenizer.py <nano>/tokenizer.json tokenizer.web.json
"""
from __future__ import annotations

import json
import sys

GPT2_SPLIT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def web_tokenizer(t: dict) -> dict:
    t = json.loads(json.dumps(t))
    assert t["pre_tokenizer"] == {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                                  "use_regex": True}, "not the tokenizer this was written for"
    t["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": GPT2_SPLIT}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]}
    t["added_tokens"] = [a for a in t["added_tokens"] if not a["special"]]
    t["post_processor"] = None
    return t


def dumps(t: dict) -> str:
    return json.dumps(t, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    src, out = sys.argv[1:3]
    with open(src, encoding="utf-8") as f:
        text = dumps(web_tokenizer(json.load(f)))
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out} ({len(text.encode())} bytes)")
