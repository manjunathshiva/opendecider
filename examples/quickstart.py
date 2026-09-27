"""OpenDecider quickstart: typed questions about one state, calibrated probabilities back.

    pip install opendecider            # nano
    pip install "opendecider[small]"   # small (adds peft)
    python examples/quickstart.py
"""
import sys

from opendecider import Choice, Noul, Score, load

# ~0.8 GB download on first use; runs on CPU, NVIDIA or Apple Silicon (a local folder works too)
model = load(sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano")

ticket = {
    "subject": "Charged twice!!",
    "body": "My card was charged twice for order #1182 and nobody answers the phone. Fix this today or I cancel.",
    "customer_tier": "gold",
}

r = model.system_one(ticket, {
    "team": Choice("Which team should handle this ticket?",
                   {"billing": "charges, invoices, refunds", "technical": "bugs, outages, errors",
                    "account": "login, profile, cancellation"}),
    "urgent": Noul("Does this need a reply today?"),
    "churn_risk": Score("How likely is the customer to leave?", ["unlikely", "possible", "likely", "very likely"]),
})

for name, a in r["answers"].items():
    top = a.get("choice", a.get("score", a.get("noul")))
    print(f"{name:11s} {a['type']:6s} -> {top}   {a['probabilities']}")
print(f"{r['latency_ms']} ms for {len(r['answers'])} questions")
