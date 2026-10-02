"""PydanticAI: OpenDecider as a toolset for agents.

    pip install "opendecider[pydantic-ai]"

    from opendecider.integrations.pydantic_ai import decision_toolset

    agent = Agent("openai:gpt-...", toolsets=[decision_toolset()])     # decide, choose, yes_no, score
    agent = Agent("openai:gpt-...", capabilities=[guardrail_capability()])         # pydantic-ai 2.x: block attacks
    agent = Agent("openai:gpt-...", history_processors=[guardrail_processor()])    # pydantic-ai 1.x

Invalid input raises `ModelRetry` with what to fix, so the model corrects its call and tries again. To route a
pydantic-graph or your own code, call `DecisionRouter` (the route name, or the fallback when unsure). The model loads on
the first call (pass a model name, or an `OpenDecider` you already loaded). Answers are the same as the MCP server's: a
probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

from typing import Any

from .. import guard as guarding
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


def _prompt_text(message) -> str | None:
    """The user's prompt in a model request, or None when the request carries none (tool results, retries)."""
    texts = []
    for part in getattr(message, "parts", None) or []:
        if getattr(part, "part_kind", None) != "user-prompt":
            continue
        content = part.content
        texts += [content] if isinstance(content, str) else [c for c in content if isinstance(c, str)]
    return "\n".join(texts) if texts else None


def guardrail_processor(guard: guarding.Guard | None = None, **settings):
    """A history processor that screens each new user prompt for jailbreaks and prompt injection before the model
    sees it, and raises GuardrailError (from `agent.run`) when it is blocked; the GuardResult goes to the guard's
    `on_decision` hooks. Requests that carry no new prompt (tool results, retries) pass unchecked.
    pydantic-ai 1.x: `Agent(..., history_processors=[guardrail_processor()])`; 2.x: `guardrail_capability()`.

    guard: an `opendecider.guard.Guard`, or its settings as keywords (checks, model, threshold, on_error, on_decision).
    """
    screen = guarding.as_guard(guard, **settings)

    async def guardrail(messages: list) -> list:
        text = _prompt_text(messages[-1]) if messages else None
        if text is not None:
            await screen.aenforce(text)
        return messages

    return guardrail


def guardrail_capability(guard: guarding.Guard | None = None, **settings):
    """`guardrail_processor` as a pydantic-ai 2.x capability: `Agent(..., capabilities=[guardrail_capability()])`."""
    try:
        from pydantic_ai.capabilities import ProcessHistory
    except ImportError as e:
        raise ImportError("capabilities need pydantic-ai 2 or later; on 1.x use "
                          "Agent(..., history_processors=[guardrail_processor()])") from e
    return ProcessHistory(guardrail_processor(guard, **settings))
