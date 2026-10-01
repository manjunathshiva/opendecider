"""CrewAI: OpenDecider as tools for crew agents, the router of a Flow, and the assigner of a crew's tasks.

    pip install "opendecider[crewai]"

    from opendecider.integrations.crewai import DecisionRouter, decision_tools

    agent = Agent(role=..., goal=..., backstory=..., tools=decision_tools())   # decide, choose, yes_no, score

    route = DecisionRouter({"billing": "charges, refunds", "tech": "bugs, outages"}, "Which team should handle this?",
                           fallback="human", min_confidence=0.6)

    class Support(Flow[TicketState]):
        @router(intake)
        def triage(self):
            return route(self.state.ticket)        # "billing", "tech" or "human": the @listen labels

    assigner = TaskAssigner([billing, engineer, sales], fallback=lead, min_confidence=0.6)
    assigner.assign_all(tasks)                         # sets each task's agent from the crew members' roles and goals
    Crew(agents=[billing, engineer, sales, lead], tasks=tasks, process=Process.sequential).kickoff()

The router picks a Flow branch, and the assigner the crew member for each task, in one forward pass, with no LLM call
(in place of a hierarchical crew's manager LLM), and both send unsure inputs to the fallback. The model loads on the
first call (pass a model name, an `opendecider serve` URL, or an `OpenDecider` you already loaded). Answers are the
same as the MCP server's: a probability for every option and a calibrated `confidence`.
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


def _task_text(task) -> str:
    """What to decide on, from a CrewAI Task (its description and expected output), a dict or text."""
    if isinstance(task, str):
        return task
    get = task.get if isinstance(task, dict) else lambda k, d=None: getattr(task, k, d)
    desc, expected = get("description") or "", get("expected_output") or ""
    if not desc and not expected:
        return str(task)
    return f"{desc}\n\nExpected output: {expected}" if expected else desc


class TaskAssigner:
    """Assigns a crew's tasks to its members, from each member's role and goal, in one forward pass per task: in place
    of a hierarchical crew's manager LLM, or of assigning every task by hand.

    agents: the crew members to choose from (CrewAI `Agent`s, or anything with `role` and `goal`).
    instructions: the assignment question.
    fallback: the member (an `Agent`, usually a lead or a person-in-the-loop) for tasks no member fits confidently;
      it need not be in `agents`. fallback / min_confidence / on_error / on_decision: as for `tools.Router`.
    """

    def __init__(self, agents: list, instructions: str = "Which crew member is best qualified to do this task?", *,
                 model: Any = core.DEFAULT_MODEL, fallback: Any = None, min_confidence: float = 0.0,
                 on_error: str = "raise", on_decision=None):
        if len(agents) < 2:
            raise ValueError("TaskAssigner needs at least 2 crew members to choose from")
        # a readable, unique label per member; the goal describes it
        roles = [str(getattr(a, "role", "") or "").strip() or f"member {i + 1}" for i, a in enumerate(agents)]
        self.members: dict[str, Any] = dict(zip(core.unique_labels(roles), agents))
        options = {label: (str(getattr(a, "goal", "") or "").strip() or None) for label, a in self.members.items()}
        fallback_label = None
        if fallback is not None:
            fallback_label = next((label for label, a in self.members.items() if a is fallback), None)
            if fallback_label is None:
                fallback_label = str(getattr(fallback, "role", "") or "fallback").strip()
                while fallback_label in self.members:
                    fallback_label += " (fallback)"
                self.members[fallback_label] = fallback
        self.router = core.Router(options, instructions, model=model, fallback=fallback_label,
                                  min_confidence=min_confidence, on_error=on_error, on_decision=on_decision)

    @property
    def last(self) -> core.Decision | None:
        """The last assignment decision (its `route` is the member's label), for logging."""
        return self.router.last

    def decide(self, task) -> core.Decision:
        """The full decision for one task; `route` is the chosen member's label (its role)."""
        return self.router.decide(_task_text(task))

    def assign(self, task):
        """The crew member for `task`, also set as `task.agent` when the task has one."""
        agent = self.members[self.decide(task).route]
        if not isinstance(task, (str, dict)):
            task.agent = agent
        return agent

    def assign_all(self, tasks: list) -> list:
        """`assign` for each task, in order; returns the members chosen."""
        return [self.assign(t) for t in tasks]

    async def aassign(self, task):
        """`assign`, with inference off the event loop."""
        return await asyncio.to_thread(self.assign, task)

