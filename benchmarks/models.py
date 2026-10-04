"""Models under test. Each exposes decide(state, instructions, options, qtype) -> {option: probability}.

  opendecider  OpenDecider through the pip package (a Hub name or a local folder), or onnx:<file>, one of
               @opendecider/web's ONNX builds in native ONNX Runtime (the browser's WebAssembly gives the same logits)
  laya         Laya through its pip package (`pip install laya`), a named checkpoint
  jev          TypeSafe Jev through TypeSafe's own API (set TYPESAFE_API_KEY)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


NANO_REVISION = "beeeb640f3333aec4ef78ed3b7bd0605f973a590"   # opendecider-nano, the weights the ONNX builds come from


class OnnxNano:
    """opendecider-nano's ONNX build (manjunathshiva/opendecider-nano-ONNX, what @opendecider/web runs) in native ONNX
    Runtime on the CPU, with the Python package's prompt. Needs `pip install onnxruntime`."""

    def __init__(self, path: str):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(snapshot_download("manjunathshiva/opendecider-nano",
                                                                   revision=NANO_REVISION))
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def decide_many(self, items, info=None):
        import numpy as np
        from opendecider.prompt import nano_ids
        built = [nano_ids(self.tok, *it, 2048) for it in items]
        if info is not None:
            info.extend({"input_tokens": len(ids), "truncated": tr} for ids, tr in built)
        out = [None] * len(items)
        order = sorted(range(len(items)), key=lambda i: len(built[i][0]))
        for s in range(0, len(order), 16):   # by length, so padding stays small
            grp = order[s:s + 16]
            L = max(len(built[i][0]) for i in grp)
            x = np.array([built[i][0] + [self.tok.pad_token_id] * (L - len(built[i][0])) for i in grp], dtype=np.int64)
            a = np.array([[1] * len(built[i][0]) + [0] * (L - len(built[i][0])) for i in grp], dtype=np.int64)
            logits = self.sess.run(["logits"], {"input_ids": x, "attention_mask": a})[0]
            for r, i in enumerate(grp):
                z = logits[r][: len(built[i][0])][np.array(built[i][0]) == self.tok.mask_token_id].astype(np.float64)
                p = np.exp(z - z.max())
                out[i] = dict(zip(items[i][2], (p / p.sum()).tolist()))
        return out


def load(name: str, device: str | None = None):
    """opendecider.load, plus onnx:<file> for an ONNX build of opendecider-nano."""
    import opendecider
    if name.startswith("onnx:"):
        return opendecider.OpenDecider(OnnxNano(name[len("onnx:"):]), {"name": "opendecider-nano", "kind": "nano"})
    return opendecider.load(name, device=device)


class OpenDecider:
    def __init__(self, name: str, device: str | None = None):
        self.m = load(name, device)

    def decide(self, state, instructions, options, qtype):
        return self.m.impl.decide_many([(state, instructions, options)])[0]


class Laya:
    """Same typed question as Jev gets; Laya's answer format matches Jev's."""

    def __init__(self, checkpoint: str = "english"):
        from laya import Router
        self.router, self.checkpoint = Router(), checkpoint

    def decide(self, state, instructions, options, qtype):
        if qtype == "noul":
            q = {"type": "noul", "instructions": instructions}
        elif qtype == "score":
            q = {"type": "score", "instructions": instructions, "criteria": [options[k] or k for k in options]}
        else:
            q = {"type": "choice", "instructions": instructions, "criteria": options}
        a = self.router.predict(state, {"q": q}, model=self.checkpoint)["answers"]["q"]
        return _from_native(a, options)


class Jev:
    """TypeSafe's own API (never a reseller), questions in their native typed form."""
    URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, model: str = "jev-1.13.0"):
        self.key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not self.key:
            raise SystemExit("set TYPESAFE_API_KEY to benchmark Jev")
        self.model = model

    def decide(self, state, instructions, options, qtype):
        names = list(options)
        if qtype == "noul":
            q = {"type": "noul", "instructions": instructions}
        elif qtype == "score":
            q = {"type": "score", "instructions": instructions, "criteria": [options[k] or k for k in names]}
        else:
            q = {"type": "choice", "instructions": instructions, "criteria": options}
        body = json.dumps({"model": self.model, "state": state, "questions": {"q": q}}).encode()
        delay = 1.0
        for attempt in range(4):
            req = urllib.request.Request(self.URL, data=body, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return _from_native(json.loads(r.read())["answers"]["q"], options)
            except urllib.error.HTTPError as e:
                if e.code in (429, 529) and attempt < 3:
                    time.sleep(delay); delay *= 2
                    continue
                raise
        raise RuntimeError("Jev did not answer after 4 attempts")   # not reached: each attempt returns or raises


def _from_native(a: dict, options: dict) -> dict:
    """A Jev/Laya-style typed answer -> {option: probability} over our option names."""
    names = list(options)
    if a["type"] == "noul":
        yes = next((k for k in names if k in ("yes", "true")), names[0])
        no = next((k for k in names if k in ("no", "false")), names[-1])
        return {yes: a["noul"], no: 1 - a["noul"]}
    if a["type"] == "score":
        return {names[int(k)]: v for k, v in a["probabilities"].items()}
    return {k: a["probabilities"].get(k, 0.0) for k in names}


def get(name: str):
    if name == "jev":
        return Jev()
    if name.startswith("laya"):
        return Laya({"laya": "english", "laya-td": "typed-decisions", "laya-ml": "multilingual"}[name])
    if name.startswith("opendecider-") and "/" not in name:   # a published model, e.g. opendecider-medium-td
        return OpenDecider(f"manjunathshiva/{name}")
    return OpenDecider(name)   # a local folder or any Hub repo with opendecider.json
