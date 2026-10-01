"""Use OpenDecider with Strands Agents: route each ticket to a specialist agent, unsure ones to a person, and the
same decisions as tools an agent can call.

    pip install "opendecider[strands]"
    python examples/agent_frameworks/strands_agent.py

`DecisionRouter` picks the specialist in one forward pass, with no LLM call; below `min_confidence` it returns the
fallback instead of a likely-wrong specialist. The specialists here only report their name, so the script runs
without an API key or a cloud model: call your agents (`AGENTS[name](ticket)`). The tools are called directly here,
as an agent would call them; give them to one with `Agent(model=..., tools=decision_tools())`.
"""
import sys

from opendecider.integrations.strands import DecisionRouter, decision_tools

MODEL = sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano"
route = DecisionRouter(
    {"billing_agent": "invoices, payment methods, duplicate charges, refunds",
     "tech_support": "system errors, bugs, API downtime, stack traces",
     "sales_agent": "pricing plans, new contracts, demo requests"},
    "Which specialist agent should answer this user query?",
    model=MODEL, fallback="human_agent", min_confidence=0.6)

TICKETS = [
    "Our API has returned 500 errors since 9am and every request fails.",
    "I was charged twice for the March invoice. Please refund the duplicate.",
    "Can I get a quote for the enterprise plan for 200 seats?",
    "Hmm, not sure, something feels off.",
]
for ticket in TICKETS:
    handled_by = route(ticket)
    top = route.last
    print(f"{ticket[:60]:<60} -> {handled_by:<13} (top: {top.choice}, p = {top.confidence:.2f})")

tools = {t.tool_name: t for t in decision_tools(MODEL)}   # the same model, loaded once
print("yes_no:", tools["yes_no"](state=TICKETS[1], question="Does the customer ask for money back?"))
print("score: ", tools["score"](state=TICKETS[0], question="How urgent is this?",
                                levels=["not urgent", "soon", "blocking"]))
print("invalid call ->", tools["choose"](state=TICKETS[0], question="Which team?", options=["billing"]))
