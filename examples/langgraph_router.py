"""Route a LangGraph support graph with OpenDecider: each ticket goes to a specialist, unsure ones to a person.

    pip install "opendecider[langchain]" langgraph
    python examples/langgraph_router.py                                       # opendecider-nano: CPU, NVIDIA, Mac
    python examples/langgraph_router.py manjunathshiva/opendecider-small-td   # pip install "opendecider[small]"

`DecisionRouter` is the graph's conditional edge: one forward pass picks the next node, with no LLM call. When the top
route's probability is below `min_confidence`, the ticket goes to the fallback node instead of a likely-wrong branch.
"""
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph

from opendecider.integrations.langchain import DecisionRouter

route = DecisionRouter(
    {"billing_agent": "invoices, payment methods, duplicate charges, refunds",
     "tech_support": "system errors, bugs, API downtime, stack traces",
     "sales_agent": "pricing plans, new contracts, demo requests"},
    "Which specialist agent should answer this user query?",
    model=sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano",
    state_key="ticket", fallback="human_agent", min_confidence=0.6)


class Ticket(TypedDict):
    ticket: str
    handled_by: str


graph = StateGraph(Ticket)
graph.add_node("triage", lambda t: {})
for node in route.path_map:                       # the three specialists, plus the fallback
    graph.add_node(node, lambda t, node=node: {"handled_by": node})   # your agents go here
    graph.add_edge(node, END)
graph.set_entry_point("triage")
graph.add_conditional_edges("triage", route, route.path_map)
app = graph.compile()

TICKETS = [
    "Our API has returned 500 errors since 9am and every request fails.",
    "I was charged twice for the March invoice. Please refund the duplicate.",
    "Can I get a quote for the enterprise plan for 200 seats?",
    "Hmm, not sure, something feels off.",
]
for ticket in TICKETS:
    out = app.invoke({"ticket": ticket})
    top = route.last
    print(f"{ticket[:60]:<60} -> {out['handled_by']:<13} (top: {top['choice']}, p = {top['confidence']:.2f})")
