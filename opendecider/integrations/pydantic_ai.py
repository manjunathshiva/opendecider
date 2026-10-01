"""PydanticAI: OpenDecider as a toolset for agents.

    pip install "opendecider[pydantic-ai]"

    from opendecider.integrations.pydantic_ai import decision_toolset

    agent = Agent("openai:gpt-...", toolsets=[decision_toolset()])     # decide, choose, yes_no, score

Invalid input raises `ModelRetry` with what to fix, so the model corrects its call and tries again. To route a
pydantic-graph or your own code, call `DecisionRouter` (the route name, or the fallback when unsure). The model loads on
the first call (pass a model name, or an `OpenDecider` you already loaded). Answers are the same as the MCP server's: a
probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

from typing import Any

from .. import tools as core

try:
    from pydantic_ai import ModelRetry, Tool
    from pydantic_ai.toolsets import FunctionToolset
except ImportError as e:   # pragma: no cover
    raise ImportError('the PydanticAI integration needs: pip install "opendecider[pydantic-ai]"') from e

DecisionRouter = core.Router   # route names for your own code or a graph's nodes


def decision_toolset(model: Any = core.DEFAULT_MODEL) -> FunctionToolset:
    """A `FunctionToolset` with `decide`, `choose`, `yes_no` and `score`, sharing one model.

    model: a Hub name or local folder, a served model (`ollama:...`, `lmstudio:...`), or a loaded `OpenDecider`.
    Invalid input raises `ModelRetry`, so the model sees what to fix and retries; a model that cannot load raises
    `ModelError`, which is not something a retry fixes.
    """
    decider = core.shared(model)

    def retrying(fn, *args) -> dict[str, Any]:
        try:
            return fn(decider, *args)
        except ValueError as e:   # the caller's input: tell the model what to fix
            raise ModelRetry(str(e)) from None

    def decide(state: str | dict | list, questions: dict[str, dict]) -> dict[str, Any]:
        return retrying(core.decide, state, questions)

    def choose(state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> dict[str, Any]:
        return retrying(core.choose, state, question, options)

    def yes_no(state: str | dict | list, question: str) -> dict[str, Any]:
        return retrying(core.yes_no, state, question)

    def score(state: str | dict | list, question: str, levels: list[str]) -> dict[str, Any]:
        return retrying(core.score, state, question, levels)

    return FunctionToolset([Tool(fn, takes_ctx=False, name=fn.__name__, description=core.DESCRIPTIONS[fn.__name__])
                            for fn in (decide, choose, yes_no, score)])
