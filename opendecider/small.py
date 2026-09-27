"""OpenDecider-small: Qwen3-4B-Instruct-2507 + a LoRA adapter, read as a classifier.

Up to 26 options: the options are lettered and one forward pass gives the probability of each
letter as the next token. More options: each option name's log-probability after the shared
prompt (label mode), batched.
"""
from __future__ import annotations

import json
import math
import string

import torch
import torch.nn.functional as F

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


class SmallModel:
    def __init__(self, adapter_path: str, base: str, device: str, device_map: str | None = None):
        from peft import PeftModel
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

    def ids(self, prompt: str) -> list[int]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        s = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        return self.tok.encode(s, add_special_tokens=False)

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

    def decide_many(self, items: list[tuple]) -> list[dict]:
        return [self.decide(*it) for it in items]
