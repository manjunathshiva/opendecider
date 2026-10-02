"""OpenDecider-small: Qwen3-4B-Instruct-2507 + a LoRA adapter, read as a classifier.

Up to 26 options: the options are lettered and one forward pass gives the probability of each
letter as the next token. More options: each option name's log-probability after the shared
prompt (label mode), batched.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .prompt import LETTERS, SYSTEM, render  # noqa: F401  (re-exported: opendecider.small.render)


class SmallModel:
    def __init__(self, adapter_path: str, base: str, device: str, device_map: str | None = None):
        try:
            from peft import PeftModel
        except ModuleNotFoundError as e:
            if e.name != "peft":   # peft is there but something it imports is not: report that as it is
                raise
            raise ImportError('opendecider-small, -small-td, -medium-td and -large-td need peft: '
                              'pip install "opendecider[small]"') from e
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(base)
        if device == "cpu":
            dtype = torch.float32
        elif device == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = torch.float16          # e.g. a Colab T4 (no bf16 tensor cores)
        else:
            dtype = torch.bfloat16
        if device_map:   # larger than one GPU (e.g. opendecider-medium): spread over all GPUs
            m = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map=device_map)
            self.device = str(m.get_input_embeddings().weight.device)
        else:
            m = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
        self.m = PeftModel.from_pretrained(m, adapter_path).merge_and_unload().eval()
        self.letters = [self.tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
        self.max_input_tokens = int(getattr(self.m.config, "max_position_embeddings", 32768))

    def ids(self, prompt: str) -> list[int]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        s = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        ids = self.tok.encode(s, add_special_tokens=False)
        if len(ids) > getattr(self, "max_input_tokens", 1 << 30):
            raise ValueError(f"input is {len(ids)} tokens, longer than this model's {self.max_input_tokens}-token context")
        self._last_tokens = len(ids)
        return ids

    @torch.no_grad()
    def decide(self, state, instructions: str, options: dict) -> dict:
        names = list(options)
        if len(names) <= len(LETTERS):
            x = torch.tensor([self.ids(render(state, instructions, options))], device=self.device)
            h = self.m.model(input_ids=x).last_hidden_state[0, -1]
            logits = self.m.lm_head(h.to(self.m.lm_head.weight.device))[self.letters[:len(names)]].float()
            return dict(zip(names, torch.softmax(logits, -1).tolist()))
        pre = self.ids(render(state, instructions, options, lettered=False))
        labs = [self.tok.encode(n, add_special_tokens=False) for n in names]
        L = len(pre) + max(len(t) for t in labs)
        pad = self.tok.pad_token_id or 0
        rows = [pre + t + [pad] * (L - len(pre) - len(t)) for t in labs]
        att = [[1] * (len(pre) + len(t)) + [0] * (L - len(pre) - len(t)) for t in labs]
        scores = []
        for i in range(0, len(rows), 16):
            x = torch.tensor(rows[i:i + 16], device=self.device)
            a = torch.tensor(att[i:i + 16], device=self.device)
            h = self.m.model(input_ids=x, attention_mask=a).last_hidden_state
            for r, t in enumerate(labs[i:i + 16]):
                pos = torch.arange(len(pre) - 1, len(pre) - 1 + len(t), device=self.device)
                lp = F.log_softmax(self.m.lm_head(h[r, pos.to(h.device)].to(self.m.lm_head.weight.device)).float(), -1)
                scores.append(sum(lp[j, t[j]].item() for j in range(len(t))))
        top = max(scores)
        w = [math.exp(s - top) for s in scores]
        return {k: v / sum(w) for k, v in zip(names, w)}

    @torch.no_grad()
    def _letters_batch(self, rows: list[list[int]], n_opts: list[int]) -> list[list[float]]:
        """Lettered questions in one right-padded forward pass: the causal mask means padding after a row's
        last real token cannot change it, so each row's answer is read at its own last position."""
        L, pad = max(len(r) for r in rows), (self.tok.pad_token_id or 0)
        x = torch.tensor([r + [pad] * (L - len(r)) for r in rows], device=self.device)
        att = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows], device=self.device)
        h = self.m.model(input_ids=x, attention_mask=att).last_hidden_state
        last = torch.tensor([len(r) - 1 for r in rows], device=h.device)
        h = h[torch.arange(len(rows), device=h.device), last]
        logits = self.m.lm_head(h.to(self.m.lm_head.weight.device)).float()
        return [torch.softmax(logits[i, self.letters[:k]], -1).tolist() for i, k in enumerate(n_opts)]

    batch = 1   # questions per forward pass. 1 = the exact path every published number used; >1 trades exactness
    #             (bf16 padding changes about 1 top answer in 75) for throughput. `opendecider serve --small-batch N`.

    def decide_many(self, items: list[tuple], info: list | None = None, batch: int | None = None) -> list[dict]:
        """Up to 26 options: `batch` questions per forward pass (default: self.batch). More options: one at a time."""
        batch = batch or self.batch
        out: list = [None] * len(items)
        toks = [0] * len(items)
        lettered = [i for i, it in enumerate(items) if len(it[2]) <= len(LETTERS)]
        for j in range(0, len(lettered), batch):
            idx = lettered[j:j + batch]
            rows = [self.ids(render(*items[i])) for i in idx]
            for i, r in zip(idx, rows):
                toks[i] = len(r)
            if len(rows) == 1:
                out[idx[0]] = self.decide(*items[idx[0]])   # the exact single-question path
                toks[idx[0]] = self._last_tokens
                continue
            for i, p in zip(idx, self._letters_batch(rows, [len(items[i][2]) for i in idx])):
                out[i] = dict(zip(items[i][2], p))
        for i, it in enumerate(items):
            if out[i] is None:
                out[i] = self.decide(*it)
                toks[i] = self._last_tokens
        if info is not None:
            info.extend({"input_tokens": t, "truncated": False} for t in toks)
        return out
