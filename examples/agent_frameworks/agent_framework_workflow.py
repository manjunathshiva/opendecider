"""Route a Microsoft Agent Framework workflow with OpenDecider: a switch-case edge sends each ticket to a specialist,
unsure ones to a person.

    pip install "opendecider[agent-framework]"
    python examples/agent_frameworks/agent_framework_workflow.py

`route.cases(...)` gives the switch-case edge group's cases: one forward pass per message picks the next executor, with
no LLM call. When the top route's probability is below `min_confidence`, no case matches and the ticket goes to the
default executor. The executors here only report their name, so the script runs without an API key: use your agents'
executors. To give an agent the decisions as tools instead: `Agent(client=..., tools=decision_tools())`.
"""
import asyncio
import sys

from agent_framework import WorkflowBuilder, WorkflowContext, executor

from opendecider.integrations.agent_framework import DecisionRouter

route = DecisionRouter(
    {"billing_agent": "invoices, payment methods, duplicate charges, refunds",
     "tech_support": "system errors, bugs, API downtime, stack traces",
     "sales_agent": "pricing plans, new contracts, demo requests"},
    "Which specialist agent should answer this user query?",
    model=sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano",
    fallback="human_agent", min_confidence=0.6)


@executor(id="triage")
async def triage(ticket: str, ctx: WorkflowContext[str]) -> None:
    await ctx.send_message(ticket)


def specialist(name):   # your agents go here
    @executor(id=name)
    async def handle(ticket: str, ctx: WorkflowContext[str, str]) -> None:
        await ctx.yield_output(name)
    return handle


targets = {name: specialist(name) for name in route.names}   # the three specialists, plus the fallback
workflow = (WorkflowBuilder(start_executor=triage)
            .add_switch_case_edge_group(triage, route.cases(targets))   # the fallback's executor is the default
            .build())

TICKETS = [
    "Our API has returned 500 errors since 9am and every request fails.",
    "I was charged twice for the March invoice. Please refund the duplicate.",
    "Can I get a quote for the enterprise plan for 200 seats?",
    "Hmm, not sure, something feels off.",
]


async def main():
    for ticket in TICKETS:
        handled_by = (await workflow.run(ticket)).get_outputs()[0]
        top = route.last
        print(f"{ticket[:60]:<60} -> {handled_by:<13} (top: {top['choice']}, p = {top['confidence']:.2f})")


asyncio.run(main())
