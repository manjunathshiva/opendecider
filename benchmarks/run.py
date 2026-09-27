"""Score a model on the three benchmarks; resumable (already-answered items are skipped).

    python benchmarks/run.py --model opendecider-nano                  # all three suites
    python benchmarks/run.py --model opendecider-small --suites typed
    python benchmarks/run.py --model laya-td --suites general          # needs `pip install laya`
    TYPESAFE_API_KEY=... python benchmarks/run.py --model jev          # TypeSafe's own API
    python benchmarks/run.py --model ./my-model-folder --name mine     # any local OpenDecider folder

Writes benchmarks/results/<suite>/<name>.jsonl (ids and probabilities only, no dataset text).
Then: python benchmarks/report.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import items as I  # noqa: E402
import models as M  # noqa: E402

RESULTS = HERE / "results"


def _done(path: Path, key) -> dict:
    out = {}
    if path.exists():
        for l in path.read_text().split("\n"):
            if l.strip():
                r = json.loads(l); out[key(r)] = r
    return out


def general(model, name):
    path = RESULTS / "general" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path, lambda r: r["id"])
    with path.open("a") as f:
        for it in I.general():
            if it["id"] in done:
                continue
            t = time.perf_counter()
            p = model.decide(it["state"], it["instructions"], it["options"], it["type"])
            f.write(json.dumps({"id": it["id"], "probs": p, "wall_s": time.perf_counter() - t}) + "\n")
    print(f"general -> {path}")


def typed(model, name):
    path = RESULTS / "typed" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path, lambda r: (r["case"], r["q"]))
    with path.open("a") as f:
        for c in I.typed():
            state, qs, gold = (json.loads(c[k]) if isinstance(c[k], str) else c[k] for k in ("state", "questions", "gold"))
            for qn, q in qs.items():
                if (c["id"], qn) in done:
                    continue
                t = q["type"]
                if t == "choice":
                    opts = dict(q["criteria"])
                elif t == "score":
                    opts = {str(i): d for i, d in enumerate(q["criteria"])}
                else:   # the case's own true/false descriptions, under OpenDecider's yes/no names
                    cr = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
                    opts = {"yes": cr.get("true", "Yes"), "no": cr.get("false", "No")}
                p = model.decide(state, q["instructions"], opts, t)
                if t == "noul":
                    p = {"true": p["yes"], "false": p["no"]}
                f.write(json.dumps({"case": c["id"], "q": qn, "type": t, "workflow": c["workflow"],
                                    "pred": max(p, key=p.get), "gold": str(gold[qn]["label"]), "probs": p}) + "\n")
    print(f"typed -> {path}")


def laya_battery(model, name):
    path = RESULTS / "laya_battery" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done(path, lambda r: (r["suite"], r["i"]))
    suites, _ = I.laya_battery()
    with path.open("a") as f:
        for sname, S in suites.items():
            for i, (state, qs) in enumerate(S["cases"]):
                if (sname, i) in done:
                    continue
                (qn, q), = qs.items()
                if q["type"] == "noul":
                    cr = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
                    opts, order = {"yes": cr.get("true", "Yes"), "no": cr.get("false", "No")}, ["no", "yes"]
                else:
                    opts = dict(q["criteria"]); order = list(opts)
                p = model.decide(state, q["instructions"], opts, q["type"])
                f.write(json.dumps({"suite": sname, "i": i, "p": [float(p[k]) for k in order]}) + "\n")
    print(f"laya_battery -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="opendecider-nano | opendecider-small | laya | laya-td | jev | a folder")
    ap.add_argument("--name", default=None, help="results name (default: the model name)")
    ap.add_argument("--suites", default="general,typed,laya")
    a = ap.parse_args()
    name = a.name or Path(a.model).name
    model = M.get(a.model)
    for s in a.suites.split(","):
        {"general": general, "typed": typed, "laya": laya_battery}[s](model, name)
