"""Typed decisions as agent tools: the shared core of the MCP server and the LangChain and LlamaIndex integrations.

Four tools, the same everywhere: `decide` (any number of typed questions about one state) and the shortcuts `choose`,
`yes_no` and `score` for a single question. Answers are trimmed for agents and named plainly (`choice`, `answer` and
`probability_yes`, `level` and `label`), each with a calibrated `confidence`, so the agent can act on confident answers
and ask about the rest.

    from opendecider.tools import Decider, choose
    decider = Decider("manjunathshiva/opendecider-nano")         # loads on the first call
    choose(decider, "I was charged twice", "Which team?", {"billing": "charges", "tech": "bugs"})
"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
from typing import Any

from .questions import Choice, Noul, Score

log = logging.getLogger("opendecider.tools")

# The same bounds as `opendecider serve`, so an agent cannot send a call large enough to exhaust memory.
MAX_QUESTIONS = 64
MAX_OPTIONS = 256
MAX_STATE_CHARS = 200_000

DEFAULT_MODEL = "manjunathshiva/opendecider-nano"

DESCRIPTIONS = {
    "decide": (
        "Answer typed questions about one state, with a probability for every option.\n\n"
        "state: plain text, or any JSON (a ticket, a log record, an agent trace).\n"
        "questions: named questions, each one of\n"
        '  {"type": "choice", "instructions": "Which team?",\n'
        '   "criteria": {"billing": "charges, refunds", "tech": "bugs"}}\n'
        '  {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}   (lowest first)\n'
        '  {"type": "noul", "instructions": "Is this spam?"}                                            (yes / no)\n'
        "Returns one answer per question, each with its probabilities and a confidence."),
    "choose": (
        "Pick one option for a question about the state, with a probability for every option.\n\n"
        'options: a list of labels, or {"label": "short description"} (descriptions help with terse labels).'),
    "yes_no": "Answer a yes/no question about the state, with the probability that the answer is yes.",
    "score": "Rate the state on an ordered scale (levels lowest first), with a probability for every level.",
}


class ModelError(RuntimeError):
    """The model could not load; the message says why."""


class Decider:
    """The model behind the tools: a name (loaded on the first call) or an already-loaded `OpenDecider`.
    One inference at a time; `guard` wraps loading and inference (the MCP server uses it to keep stdout clean)."""

    def __init__(self, model: Any = DEFAULT_MODEL, loader=None, guard=None, **load_kw):
        if isinstance(model, str):
            self.name, self._model = model, None
        else:   # an OpenDecider (or anything with system_one) the caller already loaded
            self.name, self._model = getattr(model, "name", "model"), model
        self.load_kw, self._loader = load_kw, loader
        self._guard = guard or contextlib.nullcontext
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()   # torch models are not safe to call from several threads at once

    def model(self):
        with self._load_lock:
            if self._model is None:
                log.info("loading %s", self.name)
                if self._loader is None:
                    from . import load
                    self._loader = load
                try:
                    with self._guard():
                        self._model = self._loader(self.name, **self.load_kw)
                except Exception as e:   # noqa: BLE001 -- not cached: the next call tries again
                    log.exception("could not load %s", self.name)
                    raise ModelError(f"could not load model {self.name!r}: {type(e).__name__}: {e}") from None
            return self._model

    def system_one(self, state, questions: dict) -> dict:
        from . import OpenDecider
        questions = OpenDecider.prepare(questions)   # validates without a model: a bad call never triggers a download
        check(state, questions)
        model = self.model()
        with self._run_lock, self._guard():
            return model.system_one(state, questions)


def check(state, questions: dict) -> None:
    """The size limits (question types and options are validated by `OpenDecider.prepare`)."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object of named questions")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"at most {MAX_QUESTIONS} questions per call (got {len(questions)})")
    for name, q in questions.items():
        q = q.to_dict() if hasattr(q, "to_dict") else q
        crit = q.get("criteria") if isinstance(q, dict) else None
        if isinstance(crit, (list, dict)) and len(crit) > MAX_OPTIONS:
            raise ValueError(f"question {name!r}: at most {MAX_OPTIONS} options (got {len(crit)})")
        is_score = isinstance(q, dict) and q.get("type") == "score" and isinstance(crit, list)
        if is_score and len(set(map(str, crit))) < len(crit):
            raise ValueError(f"question {name!r}: score levels must be distinct")   # answers are keyed by level label
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if len(text) > MAX_STATE_CHARS:
        raise ValueError(f"the state is longer than {MAX_STATE_CHARS} characters")


def answer(a: dict) -> dict:
    """A typed answer, trimmed to what an agent needs and named plainly."""
    if a["type"] == "noul":
        out = {"answer": "yes" if a["noul"] >= 0.5 else "no", "probability_yes": round(a["noul"], 4)}
    elif a["type"] == "score":
        out = {"level": a["score"], "label": a["legend"][str(a["score"])], "expected_level": round(a["expected"], 4),
               "probabilities": {a["legend"][k]: round(v, 4) for k, v in a["probabilities"].items()}}
    else:
        out = {"choice": a["choice"], "probabilities": {k: round(v, 4) for k, v in a["probabilities"].items()}}
    out["confidence"] = round(a["confidence"], 4)
    if a.get("truncated"):
        out["truncated"] = True
    return out


def decide(decider: Decider, state, questions: dict) -> dict[str, Any]:
    r = decider.system_one(state, questions)
    out = {"model": r["model"], "answers": {k: answer(a) for k, a in r["answers"].items()}}
    if r.get("warnings"):
        out["warnings"] = r["warnings"]
    return out


def choose(decider: Decider, state, question: str, options) -> dict[str, Any]:
    r = decider.system_one(state, {question: Choice(question, options)})   # keyed by its text, so errors name it
    return answer(r["answers"][question])


def yes_no(decider: Decider, state, question: str) -> dict[str, Any]:
    r = decider.system_one(state, {question: Noul(question)})
    return answer(r["answers"][question])


def score(decider: Decider, state, question: str, levels: list) -> dict[str, Any]:
    r = decider.system_one(state, {question: Score(question, levels)})
    return answer(r["answers"][question])
