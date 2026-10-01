"""Strands Agents: OpenDecider as tools for agents.

    pip install "opendecider[strands]"

    from opendecider.integrations.strands import decision_tools

    agent = Agent(tools=decision_tools())                              # decide, choose, yes_no, score

To route between agents or steps in your own code, call `DecisionRouter` (the route name, or the fallback
when unsure). The model loads on the first call (pass a model name, or an `OpenDecider` you already loaded). Answers are
the same as the MCP server's: a probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

from typing import Any

from .. import tools as core

try:
    from strands import tool
except ImportError as e:   # pragma: no cover
    raise ImportError('the Strands integration needs: pip install "opendecider[strands]"') from e

DecisionRouter = core.Router   # route names for your own code


def decision_tools(model: Any = core.DEFAULT_MODEL) -> list:
    """Strands tools `decide`, `choose`, `yes_no` and `score`, sharing one model.

    model: a Hub name or local folder, a served model (`ollama:...`, `lmstudio:...`), or a loaded `OpenDecider`.
    Invalid input and a model that cannot load return {"error": "<what to fix>"}, so the agent can correct its call.
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
    return [tool(fn, name=fn.__name__, description=core.DESCRIPTIONS[fn.__name__]) for fn in fns]
