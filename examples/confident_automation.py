"""Automate only the decisions the model is sure about, and measure what that buys on real labels.

    pip install opendecider pandas pyarrow
    python examples/confident_automation.py                 # nano, all 400 cases (2,000 decisions)
    python examples/confident_automation.py --limit 40      # a quick look
    python examples/confident_automation.py --model manjunathshiva/opendecider-small-td   # pip install "opendecider[small]"

Runs the model over the test split of typed-decisions (https://huggingface.co/datasets/LocalLLaMA/typed-decisions:
customer service, invoice processing, security incidents and AI-agent traces, with gold labels). No OpenDecider model
was trained on this split. Then, for several confidence thresholds, it reports how many decisions would be automated
and how accurate those are; everything below the threshold goes to a person.
"""
import argparse
import json

import pandas as pd
from huggingface_hub import hf_hub_download

from opendecider import load

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="manjunathshiva/opendecider-nano")
ap.add_argument("--limit", type=int, default=0, help="first N cases per workflow (0 = all 100)")
ap.add_argument("--workflow", default="all", choices=["all", "customer_service", "invoice_processing",
                                                      "security_incidents", "agent_trace_observability"])
a = ap.parse_args()

path = hf_hub_download("LocalLLaMA/typed-decisions", f"{a.workflow}/test-00000-of-00001.parquet", repo_type="dataset")
cases = pd.read_parquet(path)
if a.limit:
    cases = cases.groupby("workflow").head(a.limit)
model = load(a.model)

js = lambda v: json.loads(v) if isinstance(v, str) else v   # the dataset stores state, questions and gold as JSON

decisions = []   # (confidence, correct)
for c in cases.itertuples():
    gold = js(c.gold)
    for q, ans in model.system_one(js(c.state), js(c.questions))["answers"].items():
        pred = ("true" if ans["noul"] >= 0.5 else "false") if ans["type"] == "noul" else str(ans[ans["type"]])
        decisions.append((ans["confidence"], pred == str(gold[q]["label"])))

n = len(decisions)
print(f"{a.model}: {len(cases)} cases, {n} decisions, accuracy {sum(ok for _, ok in decisions) / n:.3f}\n")
print("automate when p >= | automated | accuracy of automated | to a person")
for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
    auto = [ok for p, ok in decisions if p >= t]
    acc = f"{sum(auto) / len(auto):.3f}" if auto else "-"
    print(f"{t:>17.1f} | {len(auto) / n:>8.0%} | {acc:>21} | {1 - len(auto) / n:>10.0%}")

# the other way round: the most confident X% of decisions
ranked = sorted(decisions, key=lambda d: -d[0])
print("\nmost confident | accuracy")
for share in (1.0, 0.7, 0.5):
    top = ranked[:max(1, round(share * n))]
    print(f"{share:>14.0%} | {sum(ok for _, ok in top) / len(top):.3f}")
