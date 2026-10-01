"""OpenDecider as an MCP server: AI assistants and agents (Claude Code, Claude Desktop, Cursor, …) call it as a tool.

    pip install "opendecider[mcp]"
    opendecider mcp                                               # opendecider-nano over stdio
    claude mcp add opendecider -- opendecider mcp                 # register it with Claude Code

Tools: `decide` (any number of typed questions about one state), and the shortcuts `choose`, `yes_no` and `score` for a
single question. Every answer has a probability for each option, so the calling agent can act on confident answers and
ask the user about the rest. The model loads on the first call, so the client's handshake never waits for a download;
inference runs one call at a time, off the event loop.

The server speaks MCP over stdio: stdout carries the protocol, and logs and download progress go to stderr.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

from .questions import Choice, Noul, Score

log = logging.getLogger("opendecider.mcp")

# The same bounds as `opendecider serve`, so an agent cannot send a call large enough to exhaust memory.
MAX_QUESTIONS = 64
MAX_OPTIONS = 256
MAX_STATE_CHARS = 200_000

INSTRUCTIONS = (
    "OpenDecider answers typed questions about a state (text or JSON) with a calibrated probability for every option. "
    "Use it for classification, routing, triage, yes/no checks and ratings instead of reasoning them out in text. "
    "`confidence` is the probability of the top answer: act on confident answers, and ask the user when it is low. "
    "Give each option a short description when the labels alone are terse."
)


class Decider:
    """The model behind the tools: loaded on first use, one inference at a time."""

    def __init__(self, model: str = "manjunathshiva/opendecider-nano", loader=None, **load_kw):
        self.name, self.load_kw = model, load_kw
        self._loader = loader
        self._model = None
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
                    self._model = self._loader(self.name, **self.load_kw)
                except Exception as e:   # noqa: BLE001 -- not cached: the next call tries again
                    log.exception("could not load %s", self.name)
                    raise ModelError(f"could not load model {self.name!r}: {type(e).__name__}: {e}") from None
            return self._model

    def system_one(self, state, questions: dict) -> dict:
        _check(state, questions)
        model = self.model()
        with self._run_lock:
            return model.system_one(state, questions)


class ModelError(RuntimeError):
    """The model could not load or answer; the message says why (MCP runs locally, for the user who started it)."""


def _check(state, questions: dict) -> None:
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


def _answer(a: dict) -> dict:
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


def build_server(decider: Decider):
    """The MCP server with the four tools, around `decider` (tests pass a fake model through it)."""
    import anyio
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    from . import __version__

    server = MCPServer(name="opendecider", title="OpenDecider", version=__version__, instructions=INSTRUCTIONS,
                       website_url="https://manjunathshiva.github.io/opendecider/")
    pure = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

    async def ask(state, questions: dict) -> dict:   # off the event loop: inference blocks
        try:
            return await anyio.to_thread.run_sync(decider.system_one, state, questions)
        except (ValueError, ModelError) as e:   # the caller's input, or why the model is unavailable
            raise ToolError(str(e)) from None
        except Exception as e:   # noqa: BLE001 -- an inference failure: say what failed; the traceback goes to stderr
            log.exception("inference failed")
            raise ToolError(f"inference failed: {type(e).__name__}: {e}") from None

    @server.tool(annotations=pure)
    async def decide(state: str | dict | list, questions: dict[str, dict]) -> dict[str, Any]:
        """Answer typed questions about one state, with a probability for every option.

        state: plain text, or any JSON (a ticket, a log record, an agent trace).
        questions: named questions, each one of
          {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges, refunds", "tech": "bugs"}}
          {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}   (lowest first)
          {"type": "noul", "instructions": "Is this spam?"}                                            (yes / no)
        Returns one answer per question, each with its probabilities and a confidence.
        """
        r = await ask(state, questions)
        out = {"model": r["model"], "answers": {k: _answer(a) for k, a in r["answers"].items()}}
        if r.get("warnings"):
            out["warnings"] = r["warnings"]
        return out

    @server.tool(annotations=pure)
    async def choose(state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> dict[str, Any]:
        """Pick one option for a question about the state, with a probability for every option.

        options: a list of labels, or {"label": "short description"} (descriptions help with terse labels).
        """
        r = await ask(state, {question: Choice(question, options)})   # keyed by its text, so errors name it
        return _answer(r["answers"][question])

    @server.tool(annotations=pure)
    async def yes_no(state: str | dict | list, question: str) -> dict[str, Any]:
        """Answer a yes/no question about the state, with the probability that the answer is yes."""
        r = await ask(state, {question: Noul(question)})
        return _answer(r["answers"][question])

    @server.tool(annotations=pure)
    async def score(state: str | dict | list, question: str, levels: list[str]) -> dict[str, Any]:
        """Rate the state on an ordered scale (levels lowest first), with a probability for every level."""
        r = await ask(state, {question: Score(question, levels)})
        return _answer(r["answers"][question])

    return server


def run(model: str = "manjunathshiva/opendecider-nano", **load_kw) -> None:
    # stderr only (stdout carries the protocol); our own messages at INFO, other libraries' (HTTP requests) at WARNING
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("opendecider").setLevel(logging.INFO)
    build_server(Decider(model, **load_kw)).run("stdio")
