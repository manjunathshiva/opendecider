"""Agno: OpenDecider as a toolkit for agents, and as the selector of a workflow Router.

    pip install "opendecider[agno]"

    from opendecider.integrations.agno import DecisionRouter, decision_toolkit

    agent = Agent(model=..., tools=[decision_toolkit()])        # decide, choose, yes_no, score

    route = DecisionRouter({"billing": "charges, refunds", "tech": "bugs, outages"}, "Which team should handle this?",
                           fallback="human", min_confidence=0.6)
    workflow = Workflow(steps=[Router(choices=[billing, tech, human],
                                      selector=route.selector({"billing": billing, "tech": tech, "human": human}))])

The router picks a workflow step in one forward pass, with no LLM call, and sends unsure inputs to the fallback step.
The model loads on the first call (pass a model name, or an `OpenDecider` you already loaded). Answers are the same as
the MCP server's: a probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

from typing import Any, Callable

from .. import tools as core


def decision_toolkit(model: Any = core.DEFAULT_MODEL, name: str = "opendecider"):
    """An Agno `Toolkit` with `decide`, `choose`, `yes_no` and `score`, sharing one model, plus instructions telling
    the agent how to use the probabilities.

    model: a Hub name or local folder, a served model (`ollama:...`, `lmstudio:...`), or a loaded `OpenDecider`.
    Invalid input raises ValueError and a model that cannot load raises ModelError, each saying why; Agno returns
    the message to the model as the tool's result.
    """
    try:
        from agno.tools import Toolkit
    except ImportError as e:   # pragma: no cover
        raise ImportError('the Agno integration needs: pip install "opendecider[agno]"') from e

    decider = core.shared(model)

    def decide(state: str | dict | list, questions: dict[str, dict]) -> dict[str, Any]:
        return core.decide(decider, state, questions)

    def choose(state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> dict[str, Any]:
        return core.choose(decider, state, question, options)

    def yes_no(state: str | dict | list, question: str) -> dict[str, Any]:
        return core.yes_no(decider, state, question)

    def score(state: str | dict | list, question: str, levels: list[str]) -> dict[str, Any]:
        return core.score(decider, state, question, levels)

    fns = [decide, choose, yes_no, score]
    for fn in fns:   # Agno describes each tool from its docstring
        fn.__doc__ = core.DESCRIPTIONS[fn.__name__]
    return Toolkit(name=name, tools=fns, instructions=core.INSTRUCTIONS, add_instructions=True)


class DecisionRouter(core.Router):
    """Routes an Agno workflow: `selector(steps)` gives the `selector` of a `Router` step.

    routes: the route names, as a list or as {"route": "when to take it"} (descriptions help).
    instructions: the routing question, e.g. "Which team should handle this request?".
    fallback / min_confidence: take the `fallback` route when the top route's probability is below `min_confidence`.
    state: what to decide on, from the step input: "input" (the workflow's input, the default) or "previous" (the
      previous step's content); or any function of the `StepInput`.
    """

    def __init__(self, routes: dict[str, str] | list[str], instructions: str, *, model: Any = core.DEFAULT_MODEL,
                 fallback: str | None = None, min_confidence: float = 0.0, state: str | Callable = "input",
                 on_error: str = "raise", on_decision=None):
        super().__init__(routes, instructions, model=model, fallback=fallback, min_confidence=min_confidence,
                         on_error=on_error, on_decision=on_decision)
        if not callable(state) and state not in ("input", "previous"):
            raise ValueError('state must be "input", "previous" or a function of the StepInput')
        self.state = state

    def _state(self, step_input):
        if callable(self.state):
            return self.state(step_input)
        value = step_input.input if self.state == "input" else step_input.previous_step_content
        if value is None:
            raise ValueError(f'the step input has no {"input" if self.state == "input" else "previous step content"} '
                             "to route on")
        if hasattr(value, "model_dump"):   # a pydantic input
            return value.model_dump()
        return value if isinstance(value, (str, dict, list)) else str(value)

    def selector(self, steps: dict[str, Any]) -> Callable:
        """The `selector` for `Router(choices=..., selector=...)`: maps each route name to its step."""
        missing = [n for n in self.names if n not in steps]
        if missing:
            raise ValueError(f"no step for route(s) {missing}")

        def select(step_input) -> list:
            return [steps[self.route(self._state(step_input))]]

        return select
