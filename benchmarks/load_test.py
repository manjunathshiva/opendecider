"""Load test for any /v1/systemone server (OpenDecider, Laya, Jev): ramp to N concurrent users, report
latency percentiles, throughput and errors over time, and draw the charts.

    opendecider serve --model manjunathshiva/opendecider-nano &
    python benchmarks/load_test.py --url http://localhost:8000 --users 100 --ramp 20 --hold 40 --out results/load/nano-gpu

Each simulated user sends a request, waits for the answer, and sends the next one (closed loop), so the
server is never asked for more than `--users` requests at once. The payload is one support ticket with
three typed questions (choice, score, yes/no), the shape most System One traffic has.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

import httpx

TICKETS = [
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    "The dashboard has been showing a 500 error since this morning and my whole team is blocked.",
    "Can you tell me how to add a second admin to our workspace? No rush.",
    "Your last update deleted our saved reports. This is the third time. We are evaluating other vendors.",
    "I'd like to upgrade to the annual plan. Do you offer a discount for nonprofits?",
    "Password reset emails are not arriving for any of our users.",
]
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors",
                                "sales": "plans, upgrades, pricing", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel or leave?"},
}


async def user(client, url, headers, stop_at, log, t0):
    while time.perf_counter() < stop_at:
        body = {"state": random.choice(TICKETS), "questions": QUESTIONS}
        s = time.perf_counter()
        try:
            r = await client.post(url, json=body, headers=headers)
            code = r.status_code
        except httpx.HTTPError:
            code = 0   # connection error or client timeout
        e = time.perf_counter()
        log.append((s - t0, e - t0, (e - s) * 1000, code))
        if code == 503:
            await asyncio.sleep(0.05)   # a well-behaved client backs off briefly on Retry-After


async def run(a):
    url = a.url.rstrip("/") + "/v1/systemone"
    headers = {"Authorization": f"Bearer {os.environ[a.api_key_env]}"} if os.environ.get(a.api_key_env) else {}
    log: list = []
    limits = httpx.Limits(max_connections=a.users + 10, max_keepalive_connections=a.users + 10)
    async with httpx.AsyncClient(timeout=a.timeout, limits=limits) as client:
        for _ in range(a.warmup):   # load the model's kernels before timing
            await client.post(url, json={"state": TICKETS[0], "questions": QUESTIONS}, headers=headers)
        t0 = time.perf_counter()
        stop_at = t0 + a.ramp + a.hold
        tasks = []
        for i in range(a.users):   # linear ramp: user i starts at i/N of the ramp
            tasks.append(asyncio.create_task(user(client, url, headers, stop_at, log, t0)))
            await asyncio.sleep(a.ramp / a.users)
        await asyncio.gather(*tasks)
    return log


def summarise(log, a):
    ok = [r for r in log if r[3] == 200]
    lat = sorted(r[2] for r in ok)
    pct = lambda p: lat[min(len(lat) - 1, int(p / 100 * len(lat)))] if lat else None
    dur = max(r[1] for r in log) if log else 0
    steady = [r for r in ok if r[1] >= a.ramp]   # after the ramp: every user is active
    windows = []
    for w0 in range(0, int(dur) + 1, a.window):
        rs = [r for r in log if w0 <= r[1] < w0 + a.window]
        oks = [r[2] for r in rs if r[3] == 200]
        windows.append({"t": w0 + a.window, "rps": len(oks) / a.window,
                        "p50_ms": statistics.median(oks) if oks else None,
                        "errors": sum(1 for r in rs if r[3] != 200)})
    codes = {}
    for r in log:
        codes[str(r[3])] = codes.get(str(r[3]), 0) + 1
    return {"users": a.users, "ramp_s": a.ramp, "hold_s": a.hold, "requests": len(log), "ok": len(ok), "status_codes": codes,
            "error_rate": round(1 - len(ok) / len(log), 4) if log else None,
            "p50_ms": pct(50), "p95_ms": pct(95), "p99_ms": pct(99), "max_ms": lat[-1] if lat else None,
            "steady_rps": round(len(steady) / a.hold, 1) if a.hold else None,
            "steady_p50_ms": statistics.median([r[2] for r in steady]) if steady else None,
            "questions_per_request": len(QUESTIONS), "windows": windows}


def charts(summary, out: Path, label: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = [x for x in summary["windows"] if x["p50_ms"] is not None]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), dpi=130)
    ks = ["p50_ms", "p95_ms", "p99_ms", "max_ms"]
    ax[0].bar([k.split("_")[0] for k in ks], [summary[k] for k in ks], color="#2563eb")
    for i, k in enumerate(ks):
        ax[0].text(i, summary[k], f"{summary[k]:.0f}", ha="center", va="bottom", fontsize=9)
    ax[0].set_title(f"Latency percentiles ({summary['users']} users)"); ax[0].set_ylabel("ms")
    ax[1].plot([x["t"] for x in w], [x["p50_ms"] for x in w], "o-", color="#2563eb")
    ax[1].set_title("Median latency over time"); ax[1].set_xlabel("seconds since start"); ax[1].set_ylabel("ms")
    ax[2].plot([x["t"] for x in w], [x["rps"] for x in w], "o-", color="#16a34a")
    ax[2].set_title("Throughput over time"); ax[2].set_xlabel("seconds since start"); ax[2].set_ylabel("requests / s")
    for x in ax[1:]:
        x.axvline(summary["ramp_s"], color="grey", ls="--", lw=0.8)
        x.grid(alpha=0.3)
    fig.suptitle(label + f"  |  {summary['questions_per_request']} questions per request, "
                 f"error rate {summary['error_rate']:.1%}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "load_test.png", facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--ramp", type=float, default=20, help="seconds to ramp from 0 to --users")
    ap.add_argument("--hold", type=float, default=40, help="seconds at full load after the ramp")
    ap.add_argument("--window", type=int, default=2, help="seconds per time bucket")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--api-key-env", default="OPENDECIDER_API_KEY")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    log = asyncio.run(run(a))
    s = summarise(log, a)
    print(json.dumps({k: v for k, v in s.items() if k != "windows"}))
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        (out / "requests.jsonl").write_text("".join(json.dumps(dict(zip(("start_s", "end_s", "ms", "code"), r))) + "\n" for r in log))
        try:
            charts(s, out, a.label or a.url)
        except ImportError:   # charts are optional; the numbers are already saved
            print("matplotlib is not installed: skipping load_test.png (pip install matplotlib)")


if __name__ == "__main__":
    main()
