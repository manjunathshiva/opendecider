"""LangChain and LangGraph: OpenDecider as tools for an agent, and as a router for a graph.

    pip install "opendecider[langchain]"

    from opendecider.integrations.langchain import DecisionRouter, decision_tools

    tools = decision_tools()                      # decide, choose, yes_no, score: give them to any tool-calling agent

    route = DecisionRouter({"billing": "charges, refunds", "tech": "bugs, outages"}, "Which team should handle this?",
                           fallback="human", min_confidence=0.6)
    graph.add_conditional_edges("triage", route)  # a LangGraph conditional edge: returns the next node's name

The model loads on the first call (pass a model name, or an `OpenDecider` you already loaded). Answers are the same as
the MCP server's: a probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from .. import tools as core


def _text(content):
    """A message's text: content blocks (multimodal or provider-specific) joined, other blocks dropped."""
    if isinstance(content, list):
        parts = [b if isinstance(b, str) else b.get("text", "") for b in content
                 if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")]
        return "\n".join(p for p in parts if p)
    return content


def decision_tools(model: Any = core.DEFAULT_MODEL) -> list:
    """LangChain tools `decide`, `choose`, `yes_no` and `score`, sharing one model.

    model: a Hub name or local folder, a served model (`ollama:...`, `lmstudio:...`), or a loaded `OpenDecider`.
    Invalid input and a model that cannot load come back to the agent as the tool's error message.
    """
    try:
        from langchain_core.tools import StructuredTool, ToolException
    except ImportError as e:   # pragma: no cover
        raise ImportError('the LangChain integration needs: pip install "opendecider[langchain]"') from e

    decider = core.shared(model)

    # typed signatures: LangChain builds each tool's argument schema from them
    def decide(state: str | dict | list, questions: dict[str, dict]) -> dict[str, Any]:
        return core.decide(decider, state, questions)

    def choose(state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> dict[str, Any]:
        return core.choose(decider, state, question, options)

    def yes_no(state: str | dict | list, question: str) -> dict[str, Any]:
        return core.yes_no(decider, state, question)

    def score(state: str | dict | list, question: str, levels: list[str]) -> dict[str, Any]:
        return core.score(decider, state, question, levels)

    def tool(fn: Callable):
        @functools.wraps(fn)
        def run(*args, **kw):
            try:
                return fn(*args, **kw)
            except (ValueError, core.ModelError) as e:   # the caller's input, or why the model is unavailable
                raise ToolException(str(e)) from None

        async def arun(*args, **kw):   # inference blocks: off the event loop
            return await asyncio.to_thread(run, *args, **kw)

        return StructuredTool.from_function(func=run, coroutine=arun, name=fn.__name__,
                                            description=core.DESCRIPTIONS[fn.__name__], handle_tool_error=True)

    return [tool(decide), tool(choose), tool(yes_no), tool(score)]


class DecisionRouter(core.Router):
    """A LangGraph conditional edge (or any router): picks the next node from `routes`, or `fallback` when unsure.

    routes: the node names, as a list or as {"node": "when to go there"} (descriptions help).
    instructions: the routing question, e.g. "Which agent should handle this request?".
    state_key: the field of the graph state to decide on; by default the last message's text when the state has
      `messages`, otherwise the whole state.
    fallback / min_confidence: route to `fallback` when the top route's probability is below `min_confidence`.
    """

    def __init__(self, routes: dict[str, str] | list[str], instructions: str, *, model: Any = core.DEFAULT_MODEL,
                 state_key: str | None = None, fallback: str | None = None, min_confidence: float = 0.0,
                 on_error: str = "raise", on_decision=None):
        super().__init__(routes, instructions, model=model, fallback=fallback, min_confidence=min_confidence,
                         on_error=on_error, on_decision=on_decision)
        self.state_key = state_key

    def _state(self, state):
        if self.state_key is not None:
            return state[self.state_key] if isinstance(state, dict) else getattr(state, self.state_key)
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if messages:
            last = messages[-1]
            return _text(last.content) if hasattr(last, "content") else last
        return state

    def __call__(self, state) -> str:
        return self.route(self._state(state))

    @property
    def path_map(self) -> list[str]:
        """Every node the router can return, for `add_conditional_edges(..., path_map=router.path_map)`."""
        return self.names
