"""Benchmark items, rebuilt from the public sources (their text is never committed here).

  general  200 items: BANKING77 (78 options), BoolQ, Yelp 1-5 stars, ChaosNLI (50 each, seed
           20260919) + AG News and DAIR Emotion (100 each, seed 20260926). Checked against
           manifests/*.json (ids, gold labels and SHA-256 of every state and question), so the
           items are byte-identical to the ones in the published results.
  typed    LocalLLaMA/typed-decisions `all/test` (400 cases, 2,000 decisions).
  laya     Laya's own application battery: research/scripts/bench_apps.py from Laya's repository,
           pinned to the commit we ran, cloned into .cache/ and called unchanged (N=400, seed 13).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
MANIFESTS = HERE / "manifests"
LAYA_REPO = "https://github.com/NandhaKishorM/laya.git"
LAYA_COMMIT = "4066d5d5fbf08b66c6757ddeedbd797bd7655bc0"
TYPED_URL = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all/test-00000-of-00001.parquet"
BANKING_CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

NLI = {"entailment": "The hypothesis is definitely true given the premise",
       "neutral": "The hypothesis might be true or false given the premise",
       "contradiction": "The hypothesis is definitely false given the premise"}
STARS = ["1 star: very negative", "2 stars: negative", "3 stars: mixed or neutral",
         "4 stars: positive", "5 stars: very positive"]
AG = {"World": "world news, politics, international affairs", "Sports": "sports and athletes",
      "Business": "business, companies, markets and the economy", "Sci/Tech": "science and technology"}
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def _sha(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _download(url: str, path: Path, tries: int = 4) -> None:
    """Download to a temporary file and rename, retrying, so a dropped connection never
    leaves a truncated file in the cache."""
    import time
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(tries):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
            return
        except (urllib.error.URLError, urllib.error.ContentTooShortError, OSError):
            tmp.unlink(missing_ok=True)
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _parquet(dataset: str, config: str, split: str) -> list[dict]:
    import pandas as pd
    path = CACHE / f"{dataset.replace('/', '__')}__{config}__{split}.parquet"
    if not path.exists():
        _download(f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}/0.parquet", path)
    return pd.read_parquet(path).to_dict("records")


def _verify(items: list, manifest: Path) -> None:
    want = {m["id"]: m for m in json.loads(manifest.read_text())["items"]}
    bad = [it["id"] for it in items if it["id"] not in want
           or _sha(it["state"]) != want[it["id"]]["state_sha256"]
           or _sha(it["instructions"]) != want[it["id"]]["instructions_sha256"]
           or it["gold"] != want[it["id"]]["gold"]]
    if bad or len(items) != len(want):
        raise SystemExit(f"{len(bad)} items differ from {manifest.name} (first: {bad[:5]})")


def _human(r) -> dict:
    """ChaosNLI's 100 human labels as a distribution. Older parquet conversions carry a
    `label_counter` dict; newer ones only the `label_count` array, in (e, n, c) order (checked
    row by row against the older file: identical)."""
    if r.get("label_counter") is not None:
        counts = {k: int(v or 0) for k, v in r["label_counter"].items()}
    else:
        counts = dict(zip(("e", "n", "c"), (int(x) for x in r["label_count"])))
    names = {"e": "entailment", "n": "neutral", "c": "contradiction"}
    return {names[k]: v / 100 for k, v in counts.items()}


def general() -> list[dict]:
    """The 200 general items plus the 200 extra items, verified against the manifests."""
    out = CACHE / "general_items.json"
    if out.exists():
        return json.loads(out.read_text())
    rng = random.Random(20260919)
    items = []
    with urllib.request.urlopen(BANKING_CSV, timeout=60) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode())))
    for r in rows:
        r["category"] = r["category"].rstrip("?")
    intents = {k: None for k in sorted({r["category"] for r in rows})}
    intents["other"] = "None of the intents above"
    for i, r in enumerate(rng.sample(rows, 50)):
        items.append({"id": f"route-{i:02d}", "task": "route", "type": "choice", "state": {"message": r["text"]},
                      "instructions": "Which intent does the bank customer's `message` express?",
                      "options": intents, "gold": r["category"]})
    rows = _parquet("google/boolq", "default", "validation")
    bq = rng.sample([r for r in rows if r["answer"]], 25) + rng.sample([r for r in rows if not r["answer"]], 25)
    rng.shuffle(bq)
    for i, r in enumerate(bq):
        q = r["question"].strip().rstrip("?") + "?"
        items.append({"id": f"yesno-{i:02d}", "task": "yesno", "type": "noul", "state": {"passage": r["passage"]},
                      "instructions": f"According to `passage`, is the answer to this question yes? {q}",
                      "options": {"yes": "Yes", "no": "No"}, "gold": "yes" if bool(r["answer"]) else "no"})
    stars = {s: [] for s in range(5)}
    for r in _parquet("Yelp/yelp_review_full", "yelp_review_full", "test"):
        if len(r["text"]) <= 1500:
            stars[int(r["label"])].append(r)
    yl = [x for s in range(5) for x in rng.sample(stars[s], 10)]
    rng.shuffle(yl)
    for i, r in enumerate(yl):
        items.append({"id": f"rate-{i:02d}", "task": "rate", "type": "score", "state": {"review": r["text"]},
                      "instructions": "How many stars did the author of `review` give?",
                      "options": {str(k + 1): d for k, d in enumerate(STARS)}, "gold": str(int(r["label"]) + 1)})
    rows = _parquet("metaeval/chaos-mnli-ambiguity", "default", "train")
    names = {"e": "entailment", "n": "neutral", "c": "contradiction"}
    for i, r in enumerate(rng.sample(rows, 50)):
        items.append({"id": f"ambig-{i:02d}", "task": "ambig", "type": "choice",
                      "state": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                      "instructions": "Given `premise`, is `hypothesis` true, false, or undetermined?",
                      "options": NLI, "gold": names[r["majority_label"]],
                      "human": _human(r)})
    _verify(items, MANIFESTS / "general_items_manifest.json")
    rng = random.Random(20260926)
    extra = []
    rows = _parquet("fancyzhx/ag_news", "default", "test")
    for i, r in enumerate(rng.sample(rows, 100)):
        extra.append({"id": f"agnews-{i:02d}", "task": "agnews", "type": "choice", "state": {"article": r["text"]},
                      "instructions": "Which topic is `article` about?", "options": AG, "gold": list(AG)[int(r["label"])]})
    rows = _parquet("dair-ai/emotion", "split", "test")
    for i, r in enumerate(rng.sample(rows, 100)):
        extra.append({"id": f"emotion-{i:02d}", "task": "emotion", "type": "choice", "state": {"text": r["text"]},
                      "instructions": "Which emotion does the author of `text` express?",
                      "options": {e: None for e in EMOTIONS}, "gold": EMOTIONS[int(r["label"])]})
    _verify(extra, MANIFESTS / "extra_items_manifest.json")
    # the extra items were scored with option names as their own descriptions
    extra = [dict(it, options={k: (v or k) for k, v in it["options"].items()}) for it in extra]
    items += extra
    out.write_text(json.dumps(items, ensure_ascii=False))
    print(f"general: {len(items)} items rebuilt and verified against the manifests")
    return items


def typed() -> list[dict]:
    path = CACHE / "typed_decisions_test.parquet"
    if not path.exists():
        _download(TYPED_URL, path)
    import pandas as pd
    return pd.read_parquet(path).to_dict("records")


def laya_battery():
    """Laya's own battery: (suites, metrics function), from Laya's repo at the pinned commit."""
    repo = CACHE / "laya"
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "-q", LAYA_REPO, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", LAYA_COMMIT], check=True)
    sys.path[:0] = [str(repo), str(repo / "research" / "scripts")]
    import contextlib
    import bench_apps
    from bench_local import metrics
    with contextlib.redirect_stdout(io.StringIO()):
        bench_apps.build()
    return bench_apps.SUITES, metrics
