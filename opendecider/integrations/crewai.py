"""CrewAI: OpenDecider as tools for crew agents, and as the router of a Flow.

    pip install "opendecider[crewai]"

    from opendecider.integrations.crewai import DecisionRouter, decision_tools

    agent = Agent(role=..., goal=..., backstory=..., tools=decision_tools())   # decide, choose, yes_no, score

    route = DecisionRouter({"billing": "charges, refunds", "tech": "bugs, outages"}, "Which team should handle this?",
                           fallback="human", min_confidence=0.6)

    class Support(Flow[TicketState]):
        @router(intake)
        def triage(self):
            return route(self.state.ticket)        # "billing", "tech" or "human": the @listen labels

The router picks a Flow branch (or the crew member to delegate to) in one forward pass, with no LLM call, and sends
unsure inputs to the fallback. The model loads on the first call (pass a model name, or an `OpenDecider` you already
loaded). Answers are the same as the MCP server's: a probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from .. import tools as core

try:
    from crewai.tools import BaseTool
except ImportError as e:   # pragma: no cover
    raise ImportError('the CrewAI integration needs: pip install "opendecider[crewai]"') from e

_STATE = Field(description="plain text, or any JSON (a ticket, a log record, an agent trace)")


class _DecideArgs(BaseModel):
    state: str | dict | list = _STATE
    questions: dict[str, dict] = Field(description="named typed questions: choice, score or noul")


class _ChooseArgs(BaseModel):
    state: str | dict | list = _STATE
    question: str
    options: list[str] | dict[str, str] = Field(description='labels, or {"label": "short description"}')


class _YesNoArgs(BaseModel):
    state: str | dict | list = _STATE
    question: str


class _ScoreArgs(BaseModel):
    state: str | dict | list = _STATE
    question: str
    levels: list[str] = Field(description="the scale, lowest first")


class _DecisionTool(BaseTool):
    decider: Any = None
    call: Any = None

    def _run(self, **kw) -> dict:
        return core.as_result(lambda: self.call(self.decider, **kw))

    async def _arun(self, **kw) -> dict:   # async crews: inference off the event loop
        return await asyncio.to_thread(self._run, **kw)


def decision_tools(model: Any = core.DEFAULT_MODEL) -> list:
    """CrewAI tools `decide`, `choose`, `yes_no` and `score`, sharing one model.

    model: a Hub name or local folder, a served model (`ollama:...`, `lmstudio:...`), or a loaded `OpenDecider`.
    Invalid input and a model that cannot load return {"error": "<what to fix>"}, so the agent can correct its call.
    """
    decider = core.shared(model)
    specs = [("decide", core.decide, _DecideArgs), ("choose", core.choose, _ChooseArgs),
             ("yes_no", core.yes_no, _YesNoArgs), ("score", core.score, _ScoreArgs)]
    return [_DecisionTool(name=name, description=core.DESCRIPTIONS[name], args_schema=schema, decider=decider, call=fn)
            for name, fn, schema in specs]


class DecisionRouter(core.Router):
    """Routes a CrewAI Flow: call it with the state to decide on and return the result from a `@router` method.

    routes: the route names (the labels your `@listen` methods wait for), as a list or as {"route": "when to take it"}.
    instructions: the routing question, e.g. "Which team should handle this request?".
    fallback / min_confidence: return `fallback` when the top route's probability is below `min_confidence`.
    """
