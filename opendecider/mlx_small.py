"""OpenDecider-small on Apple Silicon through MLX (a merged, quantised build of the same model).

Same prompt and scoring as small.py: lettered options read from the next-token distribution;
above 26 options, each option name's log-probability after the shared prompt.
"""
from __future__ import annotations

import math
from pathlib import Path

from .prompt import LETTERS, chat_ids, render, text_ids


class MLXSmallModel:
    def __init__(self, path: str, mlx_base: str | None = None):
        """`path`: a merged MLX model, or (with `mlx_base`) an MLX LoRA adapter applied to that base at load time."""
        from mlx_lm import load
        if mlx_base:
            from huggingface_hub import snapshot_download
            base = mlx_base if Path(mlx_base).exists() else snapshot_download(mlx_base)
            self.model, self.tok = load(base, adapter_path=path)
        else:
            self.model, self.tok = load(path)
        self.letters = [self.tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
        self.device = "mlx"

    def ids(self, prompt: str) -> list[int]:
        ids = chat_ids(self.tok, prompt)
        self._last_tokens = len(ids)
        return ids

    def decide(self, state, instructions: str, options: dict) -> dict:
        import mlx.core as mx
        names = list(options)
        if len(names) <= len(LETTERS):
            x = mx.array([self.ids(render(state, instructions, options))])
            logits = self.model(x)[0, -1].astype(mx.float32)
            sel = logits[mx.array(self.letters[:len(names)])]
            p = mx.softmax(sel, axis=-1)
            return dict(zip(names, p.tolist()))
        pre = self.ids(render(state, instructions, options, lettered=False))
        labs = [text_ids(self.tok, n) for n in names]
        pad = self.tok.pad_token_id or 0
        scores = []
        for i in range(0, len(labs), 16):
            chunk = labs[i:i + 16]
            L = len(pre) + max(len(t) for t in chunk)
            # right padding only: in a causal model it never changes the label positions we read
            x = mx.array([pre + t + [pad] * (L - len(pre) - len(t)) for t in chunk])
            lp = self.model(x).astype(mx.float32)
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            for r, t in enumerate(chunk):
                pos = [len(pre) - 1 + j for j in range(len(t))]
                scores.append(sum(lp[r, p_, t[j]].item() for j, p_ in enumerate(pos)))
        top = max(scores)
        w = [math.exp(s - top) for s in scores]
        return {k: v / sum(w) for k, v in zip(names, w)}

    def decide_many(self, items: list[tuple], info: list | None = None) -> list[dict]:
        out = []
        for it in items:
            out.append(self.decide(*it))
            if info is not None:
                info.append({"input_tokens": self._last_tokens, "truncated": False})
        return out
