"""Models under test. Each exposes decide(state, instructions, options, qtype) -> {option: probability}.

  opendecider  OpenDecider through the pip package (a Hub name or a local folder)
  laya         Laya through its pip package (`pip install laya`), a named checkpoint
  jev          TypeSafe Jev through TypeSafe's own API (set TYPESAFE_API_KEY)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class OpenDecider:
    def __init__(self, name: str, device: str | None = None):
        import opendecider
        self.m = opendecider.load(name, device=device)

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
    if name in ("opendecider-nano", "opendecider-small"):
        return OpenDecider(f"manjunathshiva/{name}")
    return OpenDecider(name)   # a local folder or any Hub repo with opendecider.json
