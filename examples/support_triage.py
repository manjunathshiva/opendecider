"""Support triage: route a queue of tickets, flag churn risk, and send the unsure ones to a person.

    pip install opendecider
    python examples/support_triage.py                                      # nano: CPU, NVIDIA or Apple Silicon
    python examples/support_triage.py manjunathshiva/opendecider-small-td  # pip install "opendecider[small]"

The same three questions are asked about every ticket in one call (`system_one_batch`). Each answer comes with a
probability, so the routing rule can act on the confident answers and hand the rest to a person.
"""
import sys

from opendecider import Choice, Noul, Score, load

TICKETS = [
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    "The dashboard has been showing a 500 error since this morning and my whole team is blocked.",
    "Can you tell me how to add a second admin to our workspace? No rush.",
    "I'd like to move to the annual plan. Do you offer a discount for nonprofits?",
    {"subject": "Re: renewal", "body": "Still waiting on the quote you promised last week. Our budget closes Friday.",
     "customer_tier": "enterprise", "open_tickets": 3},
]

QUESTIONS = {
    "team": Choice("Which team should handle this ticket?",
                   {"billing": "charges, invoices, refunds", "technical": "bugs, outages, errors",
                    "account": "users, admins, login, settings", "sales": "plans, upgrades, pricing, quotes"}),
    "urgency": Score("How urgent is this?", ["can wait", "this week", "today"]),
    "churn_risk": Noul("Does the customer threaten to cancel or leave?"),
}

AUTO_ROUTE = 0.80   # route automatically only when the team answer is at least this likely

model = load(sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano")
results = model.system_one_batch(TICKETS, QUESTIONS)

for ticket, r in zip(TICKETS, results):
    team, urgency, churn = r["answers"]["team"], r["answers"]["urgency"], r["answers"]["churn_risk"]
    route = team["choice"] if team["confidence"] >= AUTO_ROUTE else "a person (unsure)"
    text = ticket if isinstance(ticket, str) else ticket["body"]
    print(text[:70] + ("..." if len(text) > 70 else ""))
    print(f"  team      {team['choice']:<10} p = {team['confidence']:.2f}  -> route to {route}")
    print(f"  urgency   {urgency['legend'][str(urgency['score'])]:<10} p = {urgency['confidence']:.2f}")
    print(f"  churn     {'yes' if churn['noul'] >= 0.5 else 'no':<10} p(yes) = {churn['noul']:.2f}"
          + ("  -> alert the account owner" if churn["noul"] >= 0.5 else ""))
