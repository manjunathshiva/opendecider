"""Models under test. Each exposes decide(state, instructions, options, qtype) -> {option: probability}.

  opendecider  OpenDecider through the pip package (a Hub name or a local folder), or onnx:<file>, one of
               @opendecider/web's ONNX builds in native ONNX Runtime (the browser's WebAssembly gives the same logits)
  laya         Laya through its pip package (`pip install laya`), a named checkpoint
  jev          TypeSafe Jev through TypeSafe's own API (set TYPESAFE_API_KEY)
  foundry      Microsoft-Decision-1 through your own Microsoft Foundry deployment (see Foundry)
"""
from __future__ import annotations

import email.utils
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
        elif qtype == "noul_criteria":   # a yes/no question sent with its true/false descriptions (Quietly's request)
            q = {"type": "noul", "instructions": instructions,
                 "criteria": {"true": options["yes"], "false": options["no"]}}
        elif qtype == "score":
            q = {"type": "score", "instructions": instructions, "criteria": [options[k] or k for k in options]}
        else:
            q = {"type": "choice", "instructions": instructions, "criteria": options}
        a = self.router.predict(state, {"q": q}, model=self.checkpoint)["answers"]["q"]
        return _from_native(a, options)


def _retry_after(value, fallback: float, cap: float = 120.0) -> float:
    """Seconds to wait from a Retry-After header (seconds or an HTTP date), never less than our own backoff and never
    more than `cap`; an unreadable value falls back to the backoff."""
    if not value:
        return fallback
    try:
        wait = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):   # not a date, or a date with an impossible field (hour 25)
            return fallback
        if when.tzinfo is None:
            return fallback
        wait = when.timestamp() - time.time()
    return min(max(fallback, wait), cap)


class SystemOne:
    """A /v1/systemone API (Jev's typed questions and answers): POST {model, state, questions} with a Bearer key."""
    RETRY, ATTEMPTS, TIMEOUT = (429, 529), 4, 60

    def __init__(self, url: str, key: str, model: str):
        self.URL, self.key, self.model = url, key, model

    def decide(self, state, instructions, options, qtype):
        names = list(options)
        if qtype == "noul":
            q = {"type": "noul", "instructions": instructions}
        elif qtype == "noul_criteria":   # a yes/no question sent with its true/false descriptions (Quietly's request)
            q = {"type": "noul", "instructions": instructions,
                 "criteria": {"true": options["yes"], "false": options["no"]}}
        elif qtype == "score":
            q = {"type": "score", "instructions": instructions, "criteria": [options[k] or k for k in names]}
        else:
            q = {"type": "choice", "instructions": instructions, "criteria": options}
        body = json.dumps({"model": self.model, "state": state, "questions": {"q": q}}).encode()
        delay = 1.0
        for attempt in range(self.ATTEMPTS):
            req = urllib.request.Request(self.URL, data=body, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                    return _from_native(json.loads(r.read())["answers"]["q"], options)
            except urllib.error.HTTPError as e:
                if e.code in self.RETRY and attempt < self.ATTEMPTS - 1:
                    time.sleep(_retry_after(e.headers.get("Retry-After"), delay)); delay *= 2
                    continue
                raise
            except (urllib.error.URLError, OSError):   # refused or reset connections: the request never reached the model
                if attempt < self.ATTEMPTS - 1:
                    time.sleep(delay); delay *= 2
                    continue
                raise
        raise RuntimeError(f"{self.model} did not answer after {self.ATTEMPTS} attempts")   # not reached


class Jev(SystemOne):
    """TypeSafe's own API (never a reseller), questions in their native typed form."""

    def __init__(self, model: str = "jev-1.13.0"):
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise SystemExit("set TYPESAFE_API_KEY to benchmark Jev")
        super().__init__("https://api.typesafe.ai/v1/systemone", key, model)


class Foundry(SystemOne):
    """Microsoft-Decision-1 through your own Microsoft Foundry deployment: the same typed questions and answers as Jev's
    API. Set FOUNDRY_END_POINT (…/providers/microsoft/v1/systemone), FOUNDRY_API_KEY and FOUNDRY_DEPLOYMENT (the
    deployment's name, which the request's "model" must match). FOUNDRY_API_KEY is sent as a Bearer token: the Foundry
    resource's key (what the committed runs used) or a Microsoft Entra access token for https://ai.azure.com/.default."""
    RETRY, ATTEMPTS, TIMEOUT = (429, 500, 502, 503, 504), 8, 20   # answers take ~1 s; a dropped connection should not cost 60

    def __init__(self):
        url = os.environ.get("FOUNDRY_END_POINT", "").strip()
        key = os.environ.get("FOUNDRY_API_KEY", "").strip()
        model = os.environ.get("FOUNDRY_DEPLOYMENT", "").strip()
        if not (url and key and model):
            raise SystemExit("set FOUNDRY_END_POINT, FOUNDRY_API_KEY and FOUNDRY_DEPLOYMENT to benchmark Microsoft-Decision-1")
        super().__init__(url, key, model)


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
    if name == "foundry":
        return Foundry()
    if name.startswith("laya"):
        return Laya({"laya": "english", "laya-td": "typed-decisions", "laya-ml": "multilingual"}[name])
    if name.startswith("opendecider-") and "/" not in name:   # a published model, e.g. opendecider-medium-td
        return OpenDecider(f"manjunathshiva/{name}")
    return OpenDecider(name)   # a local folder or any Hub repo with opendecider.json
