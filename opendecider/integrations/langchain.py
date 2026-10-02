"""LangChain and LangGraph: OpenDecider as tools for an agent, and as a router for a graph.

    pip install "opendecider[langchain]"

    from opendecider.integrations.langchain import DecisionRouter, decision_tools

    tools = decision_tools()                      # decide, choose, yes_no, score: give them to any tool-calling agent

    route = DecisionRouter({"billing": "charges, refunds", "tech": "bugs, outages"}, "Which team should handle this?",
                           fallback="human", min_confidence=0.6)
    graph.add_conditional_edges("triage", route)  # a LangGraph conditional edge: returns the next node's name

    chain = guardrail_runnable() | prompt | llm   # blocks jailbreaks and prompt injection (GuardrailError)

    triage = decision_runnable({"team": Choice("Which team?", {"billing": "charges", "tech": "bugs"}),
                                "urgent": Noul("Is this urgent?")})
    triage.batch(tickets)                         # typed answers for every ticket, in batched model calls
    evaluate(app, data=dataset, evaluators=[decision_evaluator("grounded", "Is `output` supported by `input`?")])
    safe_docs = retriever | guardrail_runnable(mode="filter")   # drops passages carrying injected instructions

The model loads on the first call (pass a model name, or an `OpenDecider` you already loaded). Answers are the same as
the MCP server's: a probability for every option and a calibrated `confidence`.
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from .. import guard as guarding
from .. import _install_hint
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
        raise ImportError(f"the LangChain integration needs: {_install_hint('langchain')}") from e

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
        return self._decide_from(state, self._state).route

    @property
    def path_map(self) -> list[str]:
        """Every node the router can return, for `add_conditional_edges(..., path_map=router.path_map)`."""
        return self.names


GUARD_MODES = ("raise", "annotate", "filter")


def _guard_text(value, key: str | None):
    """The text to screen in a Runnable's input: a string, a document's content, a message's (or the last human
    message's) text, a prompt value's last human message, or a dict's `key` (by default "input", "question" or the
    last of "messages")."""
    if isinstance(value, str):
        return value
    if hasattr(value, "page_content"):   # a Document
        return value.page_content
    if hasattr(value, "to_messages"):    # a PromptValue
        value = value.to_messages()
    if isinstance(value, dict):
        if key is not None:
            return _guard_text(value[key], None)
        for k in ("input", "question", "messages"):
            if k in value:
                return _guard_text(value[k], None)
        raise ValueError(f"no text to screen in a dict with keys {sorted(value)}: pass key=")
    if isinstance(value, (list, tuple)) and value and all(hasattr(m, "content") for m in value):
        human = [m for m in value if getattr(m, "type", None) == "human"] or list(value)
        return _text(human[-1].content)
    if hasattr(value, "content"):        # a message
        return _text(value.content)
    return value   # anything else is screened as given (the guard reports a non-text input as an error)


_GUARD_RUNNABLE_CLASS = None


def guardrail_runnable(guard: guarding.Guard | None = None, *, mode: str = "raise", key: str | None = None,
                       **settings):
    """A Runnable that screens its input for jailbreaks and prompt injection.

    mode: "raise" (the default) passes the input through and raises GuardrailError when it is blocked; add
      `.with_fallbacks([refusal], exceptions_to_handle=(GuardrailError,))` to the chain to answer instead.
      "annotate" returns {"input": the input, "guard": the GuardResult}, for a branch or a log.
      "filter" takes a list (retrieved documents, search results, tool outputs) and returns the items that pass.
    key: the dict key holding the text, when the input is a dict (by default "input", "question" or "messages").
    guard: an `opendecider.guard.Guard`, or its settings as keywords (checks, model, threshold, on_error, on_decision).
    `batch` screens all its inputs together.
    """
    global _GUARD_RUNNABLE_CLASS
    if mode not in GUARD_MODES:
        raise ValueError(f"mode must be one of {GUARD_MODES} (got {mode!r})")
    screen = guarding.as_guard(guard, **settings)
    if _GUARD_RUNNABLE_CLASS is None:
        try:
            from langchain_core.runnables import Runnable
        except ImportError as e:   # pragma: no cover
            raise ImportError(f"the LangChain guardrail needs: {_install_hint('langchain')}") from e

        class GuardrailRunnable(Runnable):
            """Screens its input with an OpenDecider Guard (see `guardrail_runnable`)."""

            def __init__(self, guard: guarding.Guard, mode: str, key: str | None):
                self.guard, self.mode, self.key = guard, mode, key

            def _results(self, values: list) -> list:
                return self.guard.check_many([_guard_text(v, self.key) for v in values])

            def _apply(self, value, result):
                if self.mode == "annotate":
                    return {"input": value, "guard": result}
                if not result.passed:
                    raise guarding.GuardrailError(result)
                return value

            def _run(self, value):
                if self.mode == "filter":
                    if not isinstance(value, (list, tuple)):
                        raise ValueError('mode="filter" takes a list (documents, passages, tool outputs)')
                    return [v for v, r in zip(value, self._results(list(value))) if r.passed]
                return self._apply(value, self._results([value])[0])

            def invoke(self, input, config=None, **kwargs):
                return self._call_with_config(self._run, input, config)

            async def ainvoke(self, input, config=None, **kwargs):
                return await asyncio.to_thread(self.invoke, input, config, **kwargs)

            def batch(self, inputs, config=None, *, return_exceptions: bool = False, **kwargs):
                if self.mode == "filter" or not inputs:
                    return super().batch(inputs, config, return_exceptions=return_exceptions, **kwargs)
                texts: dict = {}
                out: list = list(inputs)
                for i, value in enumerate(inputs):
                    try:
                        texts[i] = _guard_text(value, self.key)
                    except (KeyError, ValueError) as e:   # no text to screen: this input's own error
                        if not return_exceptions:
                            raise
                        out[i] = e
                results = self.guard.check_many(list(texts.values()))   # one batched screening for the rest
                for i, result in zip(texts, results):
                    try:
                        out[i] = self._apply(inputs[i], result)
                    except guarding.GuardrailError as e:
                        if not return_exceptions:
                            raise
                        out[i] = e
                return out

            async def abatch(self, inputs, config=None, *, return_exceptions: bool = False, **kwargs):
                return await asyncio.to_thread(self.batch, inputs, config, return_exceptions=return_exceptions,
                                               **kwargs)

        _GUARD_RUNNABLE_CLASS = GuardrailRunnable
    return _GUARD_RUNNABLE_CLASS(screen, mode, key)


_DECISION_RUNNABLE_CLASS = None


def _decision_state(value, key: str | None):
    if key is not None:
        return value[key] if isinstance(value, dict) else getattr(value, key)
    messages = isinstance(value, list) and value and hasattr(value[0], "content")
    if isinstance(value, (str, dict, list)) and not messages:
        return value
    return _guard_text(value, None)   # a document, a message, a prompt value or a message list: its text


def decision_runnable(questions: dict, model: Any = core.DEFAULT_MODEL, *, key: str | None = None):
    """A Runnable that answers typed questions about its input (triage, extraction, checks) in one forward pass, with
    a probability for every option. `invoke` returns {question: answer} (the answers of the `decide` tool); `batch`
    answers many inputs in batched model calls.

    questions: named typed questions, as for `decide` (dicts, or `Choice`, `Score` and `Noul`).
    key: the field of a dict input to decide on (by default the whole input: text, JSON, a document or messages).
    """
    global _DECISION_RUNNABLE_CLASS
    from .. import OpenDecider
    questions = OpenDecider.prepare(questions)   # a bad question fails here, not on the first input
    core.check("", questions)
    decider = core.shared(model)
    if _DECISION_RUNNABLE_CLASS is None:
        try:
            from langchain_core.runnables import Runnable
        except ImportError as e:   # pragma: no cover
            raise ImportError(f"the LangChain integration needs: {_install_hint('langchain')}") from e

        class DecisionRunnable(Runnable):
            """Answers typed questions about its input (see `decision_runnable`)."""

            def __init__(self, decider, questions: dict, key: str | None):
                self.decider, self.questions, self.key = decider, questions, key

            def _answer(self, value) -> dict:
                return core.decide(self.decider, _decision_state(value, self.key), self.questions)["answers"]

            def invoke(self, input, config=None, **kwargs):
                return self._call_with_config(self._answer, input, config)

            async def ainvoke(self, input, config=None, **kwargs):
                return await asyncio.to_thread(self.invoke, input, config, **kwargs)

            def batch(self, inputs, config=None, *, return_exceptions: bool = False, **kwargs):
                inputs = list(inputs)
                size = min(core.MAX_BATCH_STATES, core.MAX_BATCH_ITEMS // len(self.questions))
                out: list = []
                for b in range(0, len(inputs), size):
                    chunk = inputs[b:b + size]
                    try:
                        states = [_decision_state(v, self.key) for v in chunk]
                        answered = core.decide_batch(self.decider, states, self.questions)["results"]
                        out += [r["answers"] for r in answered]
                    except Exception:   # noqa: BLE001 -- one by one, so each input gets its own answer or error
                        if not return_exceptions and len(chunk) == 1:
                            raise
                        for v in chunk:
                            try:
                                out.append(self._answer(v))
                            except Exception as e:   # noqa: BLE001 -- returned in place, as Runnable.batch does
                                if not return_exceptions:
                                    raise
                                out.append(e)
                return out

            async def abatch(self, inputs, config=None, *, return_exceptions: bool = False, **kwargs):
                return await asyncio.to_thread(self.batch, inputs, config, return_exceptions=return_exceptions,
                                               **kwargs)

        _DECISION_RUNNABLE_CLASS = DecisionRunnable
    return _DECISION_RUNNABLE_CLASS(decider, questions, key)


def decision_evaluator(key: str, question: str, *, levels: list[str] | None = None,
                       model: Any = core.DEFAULT_MODEL) -> Callable:
    """An evaluator for LangSmith's `evaluate(..., evaluators=[...])` that scores each run with one question about its
    `input`, `output` and `reference` (the reference outputs, when the dataset has them), in one forward pass instead
    of an LLM judge. Refer to them by name in the question, e.g. "Is `output` supported by `input`?".

    levels: an ordered scale (lowest first) for a rating; without it the question is yes/no.
    Returns {"key", "score", "comment"}: the probability of yes, or the expected level scaled to 0..1 for a rating.
    """
    from .. import OpenDecider
    from ..questions import Noul, Score
    q = Noul(question) if levels is None else Score(question, list(levels))
    asked = OpenDecider.prepare({key: q})   # a bad question fails here, not on the first run
    decider = core.shared(model)

    def evaluator(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
        state = {"input": inputs, "output": outputs}
        if reference_outputs:
            state["reference"] = reference_outputs
        a = core.decide(decider, state, asked)["answers"][key]
        if levels is None:
            p = a["probability_yes"]
            return {"key": key, "score": p, "comment": f"yes with probability {p}"}
        scaled = a["expected_level"] / (len(levels) - 1)
        return {"key": key, "score": round(scaled, 4), "value": a["label"],
                "comment": f"{a['label']} with probability {a['confidence']}"}

    evaluator.__name__ = key
    return evaluator
