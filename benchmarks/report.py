"""Every table from benchmarks/results/ (plus Laya's own committed battery results and Antz AI's
published per-question CSV), with the same metric code that produced the README numbers.

    python benchmarks/report.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import random
import statistics as st
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import items as I  # noqa: E402

RESULTS = HERE / "results"
ANTZ_CSV = ("https://raw.githubusercontent.com/pavanjava/jev_and_laya_benchmarking/"
            "{ref}/results/benchmark.csv")
CORE = ["route", "yesno", "rate", "ambig"]


def norm(p):
    p = {k: max(float(v), 0.0) for k, v in p.items()}
    s = sum(p.values())
    return {k: v / s for k, v in p.items()} if s > 0 else p


def jsd(p, q):
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0) + q.get(k, 0)) for k in keys}
    kl = lambda a: sum(a.get(k, 0) * math.log2(a.get(k, 0) / m[k]) for k in keys if a.get(k, 0) > 0)
    return 0.5 * kl(p) + 0.5 * kl(q)


def ece(confs, correct, bins=10):
    total, n = 0.0, len(confs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            total += len(idx) / n * abs(st.mean(confs[i] for i in idx) - st.mean(correct[i] for i in idx))
    return total


def rows(path):
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


def general_table():
    its = {i["id"]: i for i in I.general()}
    print("\n## General decisions (200 core items + AG News / DAIR Emotion)\n")
    print("| model | route | yes/no | rating | ambig | AG News | Emotion | accuracy (200) | ECE | JSD vs humans | median ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in sorted((RESULTS / "general").glob("*.jsonl")):
        recs = {r["id"]: r for r in rows(path)}
        if len(set(recs) & set(its)) < len(its):   # unfinished run: never report a partial table row
            print(f"| {path.stem} | (incomplete: {len(set(recs) & set(its))}/{len(its)} items, skipped) |")
            continue
        acc, pooled, confs, cor = {}, [], [], []
        for t in CORE + ["agnews", "emotion"]:
            rs = [(its[i], recs[i]) for i in its if its[i]["task"] == t and i in recs]
            acc[t] = st.mean(float(max(norm(r["probs"]), key=norm(r["probs"]).get) == it["gold"]) for it, r in rs) if rs else None
            if t in CORE:
                for it, r in rs:
                    p = norm(r["probs"]); k = max(p, key=p.get)
                    confs.append(p[k]); cor.append(float(k == it["gold"])); pooled.append(float(k == it["gold"]))
        amb = [jsd(norm(recs[i]["probs"]), its[i]["human"]) for i in its if its[i]["task"] == "ambig" and i in recs]
        walls = [r["wall_s"] for r in recs.values() if r.get("wall_s")]
        ms = f"{1000 * st.median(walls):.0f}" if walls else "–"   # replayed answers carry no latency
        f = lambda x: "–" if x is None else f"{x:.2f}"
        print(f"| {path.stem} | " + " | ".join(f(acc[t]) for t in CORE + ["agnews", "emotion"]) +
              f" | **{st.mean(pooled):.3f}** | {ece(confs, cor):.3f} | {st.mean(amb):.3f} | {ms} |")


def typed_table(ref="ebeddf4f166058ac1cf094050d157bc5fed4943a"):   # Antz AI results as compared (pinned commit)
    print("\n## typed-decisions (test split, 2,000 decisions)\n")
    antz = {}
    try:
        with urllib.request.urlopen(ANTZ_CSV.format(ref=ref), timeout=60) as r:
            for x in csv.DictReader(io.StringIO(r.read().decode())):
                if x["split"] == "test":
                    antz.setdefault(x["model"], {})[(x["id"], x["question"])] = (int(x["correct"]), x["gold"], x["qtype"])
    except Exception as e:
        print(f"(Antz AI CSV unavailable: {e}; showing our models only)\n")
    table = {f"{m} (Antz AI run)": {k: v[0] for k, v in d.items()} for m, d in antz.items()}
    meta = next(iter(antz.values()), {})
    qtype = {k: v[2] for k, v in meta.items()}
    for path in sorted((RESULTS / "typed").glob("*.jsonl")):
        rs = rows(path)
        if len(rs) < 2000:   # a partial run would shrink the common question set for every model
            print(f"(skipping {path.stem}: incomplete, {len(rs)}/2000 decisions)\n")
            continue
        if meta:
            mism = sum(1 for r in rs if (r["case"], r["q"]) in meta and str(r["gold"]) != meta[(r["case"], r["q"])][1])
            if mism:
                print(f"warning: {path.stem} has {mism} gold labels that differ from Antz AI's")
        table[path.stem] = {(r["case"], r["q"]): int(str(r["pred"]) == str(r["gold"])) for r in rs}
        qtype.update({(r["case"], r["q"]): r["type"] for r in rs})
    common = set.intersection(*(set(t) for t in table.values()))
    cases = sorted({k[0] for k in common})
    by_case = {}
    for k in common:
        by_case.setdefault(k[0], []).append(k)
    rng = random.Random(0)
    boots = [[rng.choice(cases) for _ in cases] for _ in range(2000)]
    acc = lambda m, ks: sum(table[m][k] for k in ks) / len(ks)
    ref_m = "laya (Antz AI run)" if "laya (Antz AI run)" in table else None

    def ci(m):
        d = sorted(acc(m, ks) - acc(ref_m, ks) for ks in ([k for c in b for k in by_case[c]] for b in boots))
        return f"{acc(m, common) - acc(ref_m, common):+.3f} [{d[50]:+.3f}, {d[1949]:+.3f}]"

    print(f"{len(common)} decisions in {len(cases)} cases. Laya here is its typed-decisions checkpoint (fine-tuned on the train split).\n")
    print("| model | accuracy | choice | score | yes/no | vs Laya typed-decisions (95% CI, paired bootstrap) |")
    print("|---|---|---|---|---|---|")
    for m in sorted(table, key=lambda m: -acc(m, common)):
        per = [acc(m, [k for k in common if qtype.get(k) == t]) for t in ("choice", "score", "noul")]
        print(f"| {m} | **{acc(m, common):.3f}** | " + " | ".join(f"{x:.3f}" for x in per) +
              f" | {ci(m) if ref_m and m != ref_m else '–'} |")


def laya_table():
    suites, metrics = I.laya_battery()
    import numpy as np
    trained = {"ag_news", "support_triage", "email_spam", "phishing", "rag_relevance"}
    out = {}
    for path in sorted((RESULTS / "laya_battery").glob("*.jsonl")):
        got = {(r["suite"], r["i"]): r["p"] for r in rows(path)}
        need = sum(len(S["cases"]) for S in suites.values())
        if len(got) < need:
            print(f"(skipping {path.stem}: incomplete, {len(got)}/{need} battery cases)")
            continue
        out[path.stem] = {s.split(".")[-1]: metrics([(g, np.array(got[(s, i)])) for i, g in enumerate(S["gold"]) if (s, i) in got])["accuracy"]
                          for s, S in suites.items()}
    committed = json.loads((I.CACHE / "laya" / "research" / "results" / "app_benchmark_results.json").read_text())
    for ck in ("english", "typed-decisions", "multilingual"):
        out[f"laya {ck} (Laya's committed run)"] = {k.split(".")[-1]: v[ck]["accuracy"] for k, v in committed["suites"].items()}
    tasks = [s.split(".")[-1] for s in suites]
    print("\n## Laya's application battery (10 tasks × 400; * = datasets Laya trained on)\n")
    print("| model | all 10 | Laya-trained 5 | other 5 | " + " | ".join(t + ("*" if t in trained else "") for t in tasks) + " |")
    print("|---" * (4 + len(tasks)) + "|")
    for m, x in sorted(out.items(), key=lambda kv: -st.mean(kv[1].values())):
        tr = st.mean(x[t] for t in tasks if t in trained); ho = st.mean(x[t] for t in tasks if t not in trained)
        print(f"| {m} | **{st.mean(x.values()):.3f}** | {tr:.3f} | {ho:.3f} | " + " | ".join(f"{x[t]:.3f}" for t in tasks) + " |")


if __name__ == "__main__":
    general_table()
    typed_table()
    laya_table()
