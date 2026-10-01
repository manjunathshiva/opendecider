"""Route a CrewAI Flow with OpenDecider: the @router sends each ticket to a specialist crew, unsure ones to a person.

    pip install "opendecider[crewai]"
    python examples/agent_frameworks/crewai_flow.py

`route(...)` returns the label the matching `@listen` method waits for: one forward pass picks the branch, with no LLM
call. When the top route's probability is below `min_confidence`, the ticket goes to the fallback branch instead of a
likely-wrong one. The branches here only record their name, so the script runs without an API key: kick off your crews
in them. To give a crew agent the decisions as tools instead: `Agent(role=..., tools=decision_tools())`.
"""
import sys

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

from opendecider.integrations.crewai import DecisionRouter

route = DecisionRouter(
    {"billing_agent": "invoices, payment methods, duplicate charges, refunds",
     "tech_support": "system errors, bugs, API downtime, stack traces",
     "sales_agent": "pricing plans, new contracts, demo requests"},
    "Which specialist agent should answer this user query?",
    model=sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano",
    fallback="human_agent", min_confidence=0.6)


class Ticket(BaseModel):
    ticket: str = ""
    handled_by: str = ""


class Support(Flow[Ticket]):
    @start()
    def intake(self):
        pass

    @router(intake)
    def triage(self):
        return route(self.state.ticket)

    @listen("billing_agent")
    def billing(self):             # e.g. BillingCrew().crew().kickoff(inputs={"ticket": self.state.ticket})
        self.state.handled_by = "billing_agent"

    @listen("tech_support")
    def tech(self):
        self.state.handled_by = "tech_support"

    @listen("sales_agent")
    def sales(self):
        self.state.handled_by = "sales_agent"

    @listen("human_agent")
    def human(self):
        self.state.handled_by = "human_agent"


TICKETS = [
    "Our API has returned 500 errors since 9am and every request fails.",
    "I was charged twice for the March invoice. Please refund the duplicate.",
    "Can I get a quote for the enterprise plan for 200 seats?",
    "Hmm, not sure, something feels off.",
]
results = []
for ticket in TICKETS:
    flow = Support()
    flow.kickoff(inputs={"ticket": ticket})
    results.append((ticket, flow.state.handled_by, route.last))
for ticket, handled_by, top in results:   # after CrewAI's own flow logs
    print(f"{ticket[:60]:<60} -> {handled_by:<13} (top: {top.choice}, p = {top.confidence:.2f})")
