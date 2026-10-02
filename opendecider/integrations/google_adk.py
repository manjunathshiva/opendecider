"""Google ADK: OpenDecider as tools for agents, and as a router agent that hands each request to a sub-agent.

    pip install "opendecider[google-adk]"

    from opendecider.integrations.google_adk import DecisionRouterAgent, decision_tools

    agent = LlmAgent(name="assistant", model="gemini-...", tools=decision_tools())   # decide, choose, yes_no, score

    router = DecisionRouterAgent(
        name="triage", sub_agents=[billing_agent, tech_agent, human_agent],
        routes={"billing_agent": "charges, refunds", "tech_agent": "bugs, outages"},
        instructions="Which team should handle this request?", fallback="human_agent", min_confidence=0.6)

    agent = LlmAgent(name="assistant", model="gemini-...", before_agent_callback=guardrail_callback())

`DecisionRouterAgent` picks the sub-agent (by name) from the user's message in one forward pass, with no LLM call, and
hands unsure requests to the fallback. The model loads on the first call (pass a model name, or an `OpenDecider` you
already loaded). Answers are the same as the MCP server's: a probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from pydantic import ConfigDict, Field, PrivateAttr

from .. import guard as guarding
from .. import tools as core

try:
    from google.adk.agents import BaseAgent
    from google.adk.tools import FunctionTool
except ImportError as e:   # pragma: no cover
    raise ImportError('the Google ADK integration needs: pip install "opendecider[google-adk]"') from e


def decision_tools(model: Any = core.DEFAULT_MODEL) -> list:
    """ADK `FunctionTool`s `decide`, `choose`, `yes_no` and `score`, sharing one model.

    model: a Hub name or local folder, a served model (`ollama:...`, `lmstudio:...`), or a loaded `OpenDecider`.
    Invalid input and a model that cannot load return {"error": "<what to fix>"}, the ADK convention for tool errors.
    """
    decider = core.shared(model)

    def decide(state: str | dict | list, questions: dict[str, dict]) -> dict[str, Any]:
        return core.as_result(core.decide, decider, state, questions)

    def choose(state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> dict[str, Any]:
        return core.as_result(core.choose, decider, state, question, options)

    def yes_no(state: str | dict | list, question: str) -> dict[str, Any]:
        return core.as_result(core.yes_no, decider, state, question)

    def score(state: str | dict | list, question: str, levels: list[str]) -> dict[str, Any]:
        return core.as_result(core.score, decider, state, question, levels)

    fns = (decide, choose, yes_no, score)
    for fn in fns:   # ADK describes each tool from its docstring
        fn.__doc__ = core.DESCRIPTIONS[fn.__name__]
    return [FunctionTool(fn) for fn in fns]


def guardrail_callback(guard: guarding.Guard | None = None, *, message: str = guarding.BLOCKED_MESSAGE, **settings):
    """A `before_agent_callback` that screens the user's message for jailbreaks and prompt injection before the agent
    runs. A blocked message skips the agent, which answers `message` instead; the GuardResult goes to the guard's
    `on_decision` hooks. Set it on the agent that receives the user's message (the root agent).

    guard: an `opendecider.guard.Guard`, or its settings as keywords (checks, model, threshold, on_error, on_decision).
    """
    from google.genai import types
    screen = guarding.as_guard(guard, **settings)

    async def before_agent(callback_context):
        text = _user_text(callback_context)
        if (await screen.acheck(text)).passed:
            return None
        return types.Content(role="model", parts=[types.Part(text=message)])

    return before_agent


def _user_text(ctx) -> str:
    content = getattr(ctx, "user_content", None)
    parts = getattr(content, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None))


class DecisionRouterAgent(BaseAgent):
    """An ADK agent that hands each request to one of its `sub_agents`, chosen by OpenDecider from the user's message.

    routes: sub-agent names, as a list or as {"name": "when to hand over"} (descriptions help).
    instructions: the routing question, e.g. "Which team should handle this request?".
    fallback / min_confidence: hand over to the `fallback` sub-agent when the top route's probability is below
      `min_confidence`.
    decision_model: the OpenDecider model (a name, an `opendecider serve` URL, or a loaded `OpenDecider`).
    on_error / on_decision: as for `tools.Router` ("fallback" hands a failed decision's request to the fallback
      sub-agent; on_decision receives every `tools.Decision`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    routes: dict[str, str] | list[str]
    instructions: str
    fallback: str | None = None
    min_confidence: float = 0.0
    decision_model: Any = Field(default=core.DEFAULT_MODEL)
    on_error: str = "raise"
    on_decision: Any = None
    _router: core.Router = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._router = core.Router(self.routes, self.instructions, model=self.decision_model,
                                   fallback=self.fallback, min_confidence=self.min_confidence,
                                   on_error=self.on_error, on_decision=self.on_decision)
        names = {a.name for a in self.sub_agents}
        missing = [n for n in self._router.names if n not in names]
        if missing:
            raise ValueError(f"no sub-agent named {missing}")

    @property
    def last(self) -> core.Decision | None:
        """The last routing decision, for logging; concurrent sessions overwrite it."""
        return self._router.last

    async def _run_async_impl(self, ctx) -> AsyncGenerator:
        # inference blocks: off the event loop; reading the message is part of the decision (its failure policy)
        name = (await asyncio.to_thread(self._router._decide_from, ctx, _user_text)).route
        target = next(a for a in self.sub_agents if a.name == name)
        async for event in target.run_async(ctx):
            yield event
