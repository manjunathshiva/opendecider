"""Route an Agno workflow with OpenDecider: a Router step sends each ticket to a specialist, unsure ones to a person.

    pip install "opendecider[agno]"
    python examples/agent_frameworks/agno_workflow.py

`route.selector(...)` is the Router step's selector: one forward pass picks the next step, with no LLM call. When the
top route's probability is below `min_confidence`, the ticket goes to the fallback step instead of a likely-wrong one.
The steps here only report their name, so the script runs without an API key: put your Agno agents or teams in them
(`Step(name="billing_agent", agent=billing_agent)`). To give an agent the decisions as tools instead:
`Agent(model=..., tools=[decision_toolkit()])`.
"""
import sys

from agno.workflow import Router, Step, StepOutput, Workflow

from opendecider.integrations.agno import DecisionRouter

route = DecisionRouter(
    {"billing_agent": "invoices, payment methods, duplicate charges, refunds",
     "tech_support": "system errors, bugs, API downtime, stack traces",
     "sales_agent": "pricing plans, new contracts, demo requests"},
    "Which specialist agent should answer this user query?",
    model=sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano",
    fallback="human_agent", min_confidence=0.6)

steps = {name: Step(name=name, executor=lambda step_input, name=name: StepOutput(content=name))   # your agents here
         for name in route.names}                   # the three specialists, plus the fallback
workflow = Workflow(name="support", steps=[
    Router(name="triage", choices=list(steps.values()), selector=route.selector(steps))])

TICKETS = [
    "Our API has returned 500 errors since 9am and every request fails.",
    "I was charged twice for the March invoice. Please refund the duplicate.",
    "Can I get a quote for the enterprise plan for 200 seats?",
    "Hmm, not sure, something feels off.",
]
for ticket in TICKETS:
    handled_by = workflow.run(input=ticket).content
    top = route.last
    print(f"{ticket[:60]:<60} -> {handled_by:<13} (top: {top['choice']}, p = {top['confidence']:.2f})")
