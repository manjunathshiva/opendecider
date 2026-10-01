"""LlamaIndex: OpenDecider as tools for an agent, and as the selector of a RouterQueryEngine.

    pip install "opendecider[llamaindex]"

    from opendecider.integrations.llamaindex import DecisionSelector, decision_tools

    tools = decision_tools()                                  # decide, choose, yes_no, score as FunctionTools

    engine = RouterQueryEngine(selector=DecisionSelector(), query_engine_tools=[sql_tool, docs_tool])

`DecisionSelector` picks one query engine from the tools' descriptions in a single forward pass, in place of an LLM
selector. The model loads on the first call (pass a model name, or an `OpenDecider` you already loaded). Answers are the
same as the MCP server's: a probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

try:
    from llama_index.core.base.base_selector import BaseSelector, SelectorResult, SingleSelection
except ImportError as e:   # pragma: no cover
    raise ImportError('the LlamaIndex integration needs: pip install "opendecider[llamaindex]"') from e

from .. import tools as core


log = logging.getLogger("opendecider.integrations.llamaindex")


def decision_tools(model: Any = core.DEFAULT_MODEL) -> list:
    """LlamaIndex `FunctionTool`s `decide`, `choose`, `yes_no` and `score`, sharing one model.

    model: a Hub name or local folder, a served model (`ollama:...`, `lmstudio:...`), or a loaded `OpenDecider`.
    Invalid input raises ValueError and a model that cannot load raises ModelError, each saying why; LlamaIndex agents
    return the message to the model as the tool's output.
    """
    from llama_index.core.tools import FunctionTool

    decider = core.shared(model)

    # typed signatures: LlamaIndex builds each tool's argument schema from them
    def decide(state: str | dict | list, questions: dict[str, dict]) -> dict[str, Any]:
        return core.decide(decider, state, questions)

    def choose(state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> dict[str, Any]:
        return core.choose(decider, state, question, options)

    def yes_no(state: str | dict | list, question: str) -> dict[str, Any]:
        return core.yes_no(decider, state, question)

    def score(state: str | dict | list, question: str, levels: list[str]) -> dict[str, Any]:
        return core.score(decider, state, question, levels)

    def tool(fn):
        async def arun(*args, **kw):   # inference blocks: off the event loop
            return await asyncio.to_thread(fn, *args, **kw)
        return FunctionTool.from_defaults(fn=fn, async_fn=arun, name=fn.__name__,
                                          description=core.DESCRIPTIONS[fn.__name__])

    return [tool(decide), tool(choose), tool(yes_no), tool(score)]


class DecisionSelector(BaseSelector):
    """Selects one of a RouterQueryEngine's query engines (or any tools) for a query, from their names and descriptions.

    model: as for `decision_tools`. instructions: the routing question asked about the query.
    max_description_chars: each choice's description is cut to this length, so long tool descriptions cannot push the
      query itself out of the model's input (opendecider-nano reads 2,048 tokens and shortens the query first).
    The selection's `reason` gives the probability, so it shows up in LlamaIndex's traces.
    """

    def __init__(self, model: Any = core.DEFAULT_MODEL,
                 instructions: str = "Which of these sources is best suited to answer the query?",
                 max_description_chars: int = 400):
        self.decider = core.shared(model)
        self.instructions = instructions
        self.max_description_chars = max_description_chars
        self.last: dict | None = None   # the last answer, for logging

    def _select(self, choices: Sequence, query) -> SelectorResult:
        if not choices:
            raise ValueError("no choices to select from")
        labels = []
        for i, c in enumerate(choices):   # a readable, unique label per choice; the description explains it
            label = (c.name or f"option {i + 1}").strip() or f"option {i + 1}"
            labels.append(label if label not in labels else f"{label} ({i + 1})")
        if len(choices) == 1:
            return SelectorResult(selections=[SingleSelection(index=0, reason="the only choice")])
        cap = self.max_description_chars
        options = {label: (c.description or "")[:cap] or None for label, c in zip(labels, choices)}
        text = query.query_str if hasattr(query, "query_str") else str(query)
        self.last = core.choose(self.decider, text, self.instructions, options)
        if self.last.get("truncated"):
            log.warning("the query was shortened to fit the model's input; shorten the tool descriptions")
        index = labels.index(self.last["choice"])
        return SelectorResult(selections=[SingleSelection(
            index=index, reason=f"OpenDecider: {labels[index]!r} with probability {self.last['confidence']:.3f}")])

    async def _aselect(self, choices: Sequence, query) -> SelectorResult:
        return await asyncio.to_thread(self._select, choices, query)

    def _get_prompts(self) -> dict:   # no prompt to expose: the model reads the choices directly
        return {}

    def _update_prompts(self, prompts: dict) -> None:
        pass
