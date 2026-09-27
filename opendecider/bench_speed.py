"""Latency by questions per call: python -m opendecider.bench_speed manjunathshiva/opendecider-nano

One support-ticket state, 1 / 5 / 10 / 50 typed questions per `system_one` call (a mix of choice,
score and yes/no), 3 warm-up calls, then the median of 20 timed calls. Prints a markdown table.
"""
from __future__ import annotations

import json
import statistics
import sys
import time

from . import Choice, Noul, Score, default_device, load

STATE = {"subject": "Charged twice for my order",
         "body": "Hi, my card was charged twice for order #1182 (EUR 89.00 each). I emailed on Monday and nobody replied. "
                 "Please refund the duplicate today, otherwise I will cancel my subscription.",
         "customer": {"tier": "gold", "tenure_months": 26, "open_tickets": 2}}
POOL = [
    Choice("Which team should handle this ticket?", {"billing": "charges, invoices, refunds", "technical": "bugs, outages",
                                                     "account": "login, profile, cancellation", "sales": "pricing, upgrades"}),
    Noul("Does this need a reply today?"),
    Score("How upset is the customer?", ["calm", "annoyed", "angry", "furious"]),
    Noul("Does the customer threaten to cancel or leave?"),
    Choice("What does the customer want?", ["refund", "replacement", "information", "cancellation"]),
]


def questions(n):
    return {f"q{i}": POOL[i % len(POOL)] for i in range(n)}


def sync(device):
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano"
    device = default_device()
    model = load(name, device=device)
    rows = []
    for n in (1, 5, 10, 50):
        qs = questions(n)
        for _ in range(3):
            model.system_one(STATE, qs)
        ts = []
        for _ in range(20):
            sync(device); t = time.perf_counter()
            model.system_one(STATE, qs)
            sync(device); ts.append((time.perf_counter() - t) * 1000)
        med = statistics.median(ts)
        rows.append({"questions": n, "ms": round(med, 1), "ms_per_question": round(med / n, 1)})
    print(json.dumps({"model": name, "device": device, "rows": rows}))
    print(f"\n| questions per call | {name.split('/')[-1]} ({device}) |\n|---|---|")
    for r in rows:
        print(f"| {r['questions']} | {r['ms']} ms ({r['ms_per_question']} ms/question) |")


if __name__ == "__main__":
    main()
