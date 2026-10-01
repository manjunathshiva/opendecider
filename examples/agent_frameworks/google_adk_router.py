"""Route a Google ADK request with OpenDecider: a router agent hands each ticket to a specialist sub-agent, unsure ones
to a person.

    pip install "opendecider[google-adk]"
    python examples/agent_frameworks/google_adk_router.py

`DecisionRouterAgent` picks the sub-agent from the user's message in one forward pass, with no LLM call. When the top
route's probability is below `min_confidence`, the request goes to the fallback sub-agent instead of a likely-wrong
one. The sub-agents here only reply with their name, so the script runs without an API key: use your `LlmAgent`s. To
give an agent the decisions as tools instead: `LlmAgent(name=..., model=..., tools=decision_tools())`.
"""
import asyncio
import sys

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from opendecider.integrations.google_adk import DecisionRouterAgent


class Specialist(BaseAgent):
    """Stands in for an LlmAgent: replies with its own name."""

    async def _run_async_impl(self, ctx):
        yield Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=self.name)]))


router = DecisionRouterAgent(
    name="triage",
    sub_agents=[Specialist(name=n) for n in ("billing_agent", "tech_support", "sales_agent", "human_agent")],
    routes={"billing_agent": "invoices, payment methods, duplicate charges, refunds",
            "tech_support": "system errors, bugs, API downtime, stack traces",
            "sales_agent": "pricing plans, new contracts, demo requests"},
    instructions="Which specialist agent should answer this user query?",
    fallback="human_agent", min_confidence=0.6,
    decision_model=sys.argv[1] if len(sys.argv) > 1 else "manjunathshiva/opendecider-nano")

TICKETS = [
    "Our API has returned 500 errors since 9am and every request fails.",
    "I was charged twice for the March invoice. Please refund the duplicate.",
    "Can I get a quote for the enterprise plan for 200 seats?",
    "Hmm, not sure, something feels off.",
]


async def main():
    sessions = InMemorySessionService()
    runner = Runner(agent=router, app_name="support", session_service=sessions)
    for i, ticket in enumerate(TICKETS):
        await sessions.create_session(app_name="support", user_id="user", session_id=str(i))
        message = types.Content(role="user", parts=[types.Part(text=ticket)])
        async for event in runner.run_async(user_id="user", session_id=str(i), new_message=message):
            handled_by = event.author
        top = router.last
        print(f"{ticket[:60]:<60} -> {handled_by:<13} (top: {top.choice}, p = {top.confidence:.2f})")


asyncio.run(main())
