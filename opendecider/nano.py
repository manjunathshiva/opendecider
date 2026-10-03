"""OpenDecider-nano: an encoder with one [MASK] marker per option and a small MLP head.

    [CLS] question: ... [SEP] [MASK] opt1 [MASK] opt2 ... [SEP] input: <state> [SEP]

The encoder's hidden state at each marker -> MLP -> one logit per option -> softmax. All
options are read together in one pass, so a question with 78 options costs one forward.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .prompt import text_ids


class NanoModel(nn.Module):
    def __init__(self, path: str, device: str, max_len: int = 2048, dtype: str = "float32"):
        super().__init__()
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
        path = Path(path)
        self.tok = AutoTokenizer.from_pretrained(path)
        # weights are stored in bf16 to halve the download; inference runs in fp32 by default, as evaluated.
        # dtype="bfloat16" is faster on CPUs with bf16 units (e.g. Intel AMX) and on GPUs; see the README for its agreement.
        dt = getattr(torch, dtype)
        self.enc = AutoModel.from_pretrained(path, dtype=dt)
        d = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, 1))
        self.head.load_state_dict({k: v.float() for k, v in load_file(path / "head.safetensors").items()})
        self.head.to(dt)
        self.mask_id, self.max_len = self.tok.mask_token_id, max_len
        self.to(device).eval()
        self.device = device

    def build(self, state, instructions: str, options: dict) -> list[int]:
        """Token ids with one [MASK] before each option; the state is truncated first, so every
        option always survives."""
        return self._build(state, instructions, options)[0]

    def _build(self, state, instructions: str, options: dict) -> tuple[list[int], bool]:
        t = self.tok
        st = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        # every text is read as plain text: "[MASK]" written in a state must not become an option marker
        q = text_ids(t, f"question: {instructions}")
        opts = []
        for k, v in options.items():
            opts += [self.mask_id] + text_ids(t, f" {k}: {v}" if v and v != k else f" {k}")
        s = text_ids(t, f"input: {st}")
        room = max(self.max_len - len(q) - len(opts) - 4, 0)
        ids = [t.cls_token_id] + q + [t.sep_token_id] + opts + [t.sep_token_id] + s[:room] + [t.sep_token_id]
        return ids, len(s) > room

    @torch.no_grad()
    def decide_many(self, items: list[tuple], info: list | None = None) -> list[dict]:
        """[(state, instructions, options)] -> [{option: probability}], in one padded batch.
        If `info` is a list, one {"input_tokens", "truncated"} dict per item is appended to it."""
        built = [self._build(*it) for it in items]
        ids = [b for b, _ in built]
        if info is not None:
            info.extend({"input_tokens": len(b), "truncated": tr} for b, tr in built)
        L, pad = max(len(x) for x in ids), self.tok.pad_token_id
        x = torch.tensor([r + [pad] * (L - len(r)) for r in ids], device=self.device)
        att = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in ids], device=self.device)
        h = self.enc(input_ids=x, attention_mask=att).last_hidden_state
        out = []
        for r, (row, (_, _, opts)) in enumerate(zip(ids, items)):
            pos = torch.tensor([i for i, t in enumerate(row) if t == self.mask_id], device=self.device)
            if len(pos) != len(opts):   # one marker per option, always: anything else would mis-assign probabilities
                raise RuntimeError(f"expected {len(opts)} option markers, found {len(pos)}")
            p = torch.softmax(self.head(h[r, pos]).squeeze(-1).float(), -1).tolist()
            out.append(dict(zip(opts, p)))
        return out
