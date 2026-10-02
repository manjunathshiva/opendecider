"""Strands Agents: OpenDecider as tools for agents.

    pip install "opendecider[strands]"

    from opendecider.integrations.strands import decision_tools

    agent = Agent(tools=decision_tools())                              # decide, choose, yes_no, score
    agent = Agent(hooks=[guardrail_hook()])                            # block jailbreaks and prompt injection

To route between agents or steps in your own code, call `DecisionRouter` (the route name, or the fallback
when unsure). The model loads on the first call (pass a model name, or an `OpenDecider` you already loaded). Answers are
the same as the MCP server's: a probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

from typing import Any

from .. import guard as guarding
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


def _user_text(messages) -> str:
    texts = []
    for m in messages or []:
        if m.get("role") == "user":
            texts += [c["text"] for c in m.get("content", []) if isinstance(c, dict) and c.get("text")]
    return "\n".join(texts)


class GuardrailHook:
    """A Strands hook provider (`Agent(hooks=[...])`) that screens each new user message for jailbreaks and prompt
    injection before the model sees it. A blocked request is cancelled with `message` as the agent's answer; the
    GuardResult goes to the guard's `on_decision` hooks. On Strands versions without cancellation, GuardrailError is
    raised instead, and the blocked message stays in `agent.messages` until you remove it.
    """

    def __init__(self, guard: guarding.Guard, message: str = guarding.BLOCKED_MESSAGE):
        self.guard, self.message = guard, message

    def register_hooks(self, registry, **kwargs) -> None:
        from strands.hooks import BeforeInvocationEvent, MessageAddedEvent
        if "cancel" in getattr(BeforeInvocationEvent, "__dataclass_fields__", {}):
            registry.add_callback(BeforeInvocationEvent, self._before_invocation)
        else:   # older Strands: no way to cancel, so a blocked message stops the run
            registry.add_callback(MessageAddedEvent, self._message_added)

    def _before_invocation(self, event) -> None:
        text = _user_text(event.messages)
        if text and not self.guard.check(text).passed:
            event.cancel = self.message

    def _message_added(self, event) -> None:
        text = _user_text([event.message])
        if text:
            self.guard.enforce(text)


def guardrail_hook(guard: guarding.Guard | None = None, *, message: str = guarding.BLOCKED_MESSAGE,
                   **settings) -> GuardrailHook:
    """A `GuardrailHook` for `Agent(hooks=[...])`.

    guard: an `opendecider.guard.Guard`, or its settings as keywords (checks, model, threshold, on_error, on_decision).
    """
    return GuardrailHook(guarding.as_guard(guard, **settings), message)
