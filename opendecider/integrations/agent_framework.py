"""Microsoft Agent Framework: OpenDecider as tools for agents, and as the switch of a workflow.

    pip install "opendecider[agent-framework]"

    from opendecider.integrations.agent_framework import DecisionRouter, decision_tools

    agent = Agent(client=..., tools=decision_tools())                # decide, choose, yes_no, score

    route = DecisionRouter({"billing": "charges, refunds", "tech": "bugs, outages"}, "Which team should handle this?",
                           fallback="human", min_confidence=0.6)
    builder.add_switch_case_edge_group(triage, route.cases({"billing": billing, "tech": tech}, default=human))

The switch picks the next executor in one forward pass, with no LLM call, and sends unsure messages to the default
executor. The model runs once per message, however many cases there are. It loads on the first call (pass a model
name, or an `OpenDecider` you already loaded). Answers are the same as the MCP server's: a probability for every option
and a calibrated `confidence`.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from .. import tools as core

log = logging.getLogger("opendecider.integrations.agent_framework")


def decision_tools(model: Any = core.DEFAULT_MODEL) -> list:
    """Agent Framework `FunctionTool`s `decide`, `choose`, `yes_no` and `score`, sharing one model.

    model: a Hub name or local folder, a served model (`ollama:...`, `lmstudio:...`), or a loaded `OpenDecider`.
    Invalid input and a model that cannot load return {"error": "<what to fix>"}: the framework hides exception text
    from the model by default (`include_detailed_errors`), and these messages tell the agent how to correct its call.
    """
    try:
        from agent_framework import tool
    except ImportError as e:   # pragma: no cover
        raise ImportError('the Agent Framework integration needs: pip install "opendecider[agent-framework]"') from e

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
    return [tool(fn, name=fn.__name__, description=core.DESCRIPTIONS[fn.__name__]) for fn in fns]


def _text(message) -> Any:
    """What to decide on, from a workflow message: text as is; an agent's response or a chat message as its text;
    JSON as is."""
    if isinstance(message, (str, dict, list)):
        return message
    response = getattr(message, "agent_response", None)   # AgentExecutorResponse
    if response is not None and getattr(response, "text", None) is not None:
        return response.text
    if getattr(message, "text", None) is not None:          # ChatMessage, AgentRunResponse
        return message.text
    if hasattr(message, "model_dump"):                      # a pydantic message
        return message.model_dump()
    return str(message)


class DecisionRouter(core.Router):
    """Routes an Agent Framework workflow: `cases(targets, default=...)` gives the cases of a switch-case edge group.

    routes: the route names, as a list or as {"route": "when to take it"} (descriptions help).
    instructions: the routing question, e.g. "Which team should handle this request?".
    fallback / min_confidence: when the top route's probability is below `min_confidence`, no case matches and the
      message goes to the `default` executor.
    state: a function of the message giving what to decide on; by default the text of an agent response or chat
      message, or the message itself.

    Switch-case conditions are synchronous, so the decision runs on the workflow's event loop: milliseconds with
    opendecider-nano. If the decision fails (a model that cannot load), the error is logged and the message goes to
    the default executor.
    """

    def __init__(self, routes: dict[str, str] | list[str], instructions: str, *, model: Any = core.DEFAULT_MODEL,
                 fallback: str | None = None, min_confidence: float = 0.0, state: Callable | None = None,
                 on_error: str = "raise", on_decision=None):
        super().__init__(routes, instructions, model=model, fallback=fallback, min_confidence=min_confidence,
                         on_error=on_error, on_decision=on_decision)
        self.state = state or _text
        self._memo: tuple[Any, str | None] = (object(), None)   # (message, route): every case reuses one decision

    def pick(self, message) -> str | None:
        """The route for a workflow message, computed once per message; None (the default executor) when the
        decision fails."""
        seen, choice = self._memo
        if seen is message:
            return choice
        try:
            choice = self._decide_from(message, self.state).route
        except Exception as e:   # the switch would swallow it case by case; say why once, then take the default
            core._log_once(log, logging.ERROR, "routing failed, sending the message to the default executor: %s", e)
            choice = None
        self._memo = (message, choice)
        return choice

    def cases(self, targets: dict[str, Any], default: Any = None) -> list:
        """The `Case`s (one per route, by name, and one for the fallback when it has a target) and the `Default` for
        `add_switch_case_edge_group`. The default (by default the fallback's target) takes failed decisions."""
        from agent_framework import Case, Default
        missing = [n for n in self.routes if n not in targets]
        if missing:
            raise ValueError(f"no target for route(s) {missing}")
        if default is None:
            default = targets.get(self.fallback) if self.fallback is not None else None
        if default is None:
            raise ValueError("a switch-case edge group needs a default target (the fallback route's executor)")
        names = list(self.routes)
        if (self.fallback is not None and self.fallback not in names and self.fallback in targets
                and targets[self.fallback] is not default):   # the framework allows one edge per target
            names.append(self.fallback)   # low-confidence messages go to the fallback's own target
        cases = [Case(condition=lambda m, name=name: self.pick(m) == name, target=targets[name]) for name in names]
        return cases + [Default(target=default)]
