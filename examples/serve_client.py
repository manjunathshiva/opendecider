"""Call `opendecider serve` over HTTP: one state, a batch of states, and retries when the server is busy.

    pip install "opendecider[serve]"
    opendecider serve --model manjunathshiva/opendecider-nano &    # http://localhost:8000
    python examples/serve_client.py                                 # or --url https://decider.internal

Standard library only, so the same code runs in any service. The server speaks TypeSafe Jev's /v1/systemone protocol:
on the wire a score answer's `score` is the expected score and `level` the most likely level. Set OPENDECIDER_API_KEY
if the server was started with one; it is sent only over https, or over http to this machine.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--wait", type=float, default=120, help="seconds to wait for the server to be ready")
a = ap.parse_args()
KEY = os.environ.get("OPENDECIDER_API_KEY")
url = urllib.parse.urlsplit(a.url)
if url.scheme not in ("http", "https") or not url.hostname:
    raise SystemExit(f"--url must be an http or https URL, got {a.url!r}")
if KEY and url.scheme == "http" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
    raise SystemExit("refusing to send OPENDECIDER_API_KEY over plain HTTP to another host; use https")


def call(path: str, body: dict | None = None, tries: int = 5) -> dict:
    headers = {"content-type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(a.url.rstrip("/") + path, data=data, headers=headers)
            if KEY:   # unredirected: urllib never copies it onto a redirect to another host
                req.add_unredirected_header("Authorization", f"Bearer {KEY}")
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < tries - 1:   # busy: wait as long as the server asks, then retry
                time.sleep(float(e.headers.get("Retry-After") or 1))
                continue
            raise SystemExit(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')}") from None
    raise SystemExit(f"{path}: server still busy after {tries} attempts")


deadline = time.time() + a.wait
while True:   # /ready is true once the model is loaded
    try:
        if call("/ready").get("ready"):
            break
    except (OSError, SystemExit):   # not listening yet, or still loading the model
        pass
    if time.time() > deadline:
        raise SystemExit(f"{a.url} is not ready after {a.wait:g} s; is `opendecider serve` running?")
    time.sleep(2)

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel or leave?"},
}

# one state
r = call("/v1/systemone", {"state": "We were billed twice for March. Refund it today or we cancel.",
                           "questions": QUESTIONS})
ans = r["answers"]
print(f"model {r['model']}, {r['usage']['input_tokens']} input tokens")
print(f"  department {ans['department']['choice']} (p = {ans['department']['confidence']:.2f})")
print(f"  urgency    level {ans['urgency']['level']}, expected score {ans['urgency']['score']:.2f} of 2")
print(f"  churn_risk p(yes) = {ans['churn_risk']['noul']:.2f}")

# the same questions about many states in one request
states = ["The app crashes on login since the update.", "How do I download last year's invoices?",
          {"subject": "Cancel", "body": "Third outage this month. Send me the cancellation form."}]
batch = call("/v1/systemone/batch", {"states": states, "questions": QUESTIONS})
for st, res in zip(states, batch["results"]):
    text = st if isinstance(st, str) else st["body"]
    print(f"  {text[:45]:<45} -> {res['answers']['department']['choice']:<9} "
          f"churn p = {res['answers']['churn_risk']['noul']:.2f}")
