"""OpenDecider as an MCP server: AI assistants and agents (Claude Code, Claude Desktop, Cursor, …) call it as a tool.

    pip install "opendecider[mcp]"
    opendecider mcp                                               # opendecider-nano over stdio
    claude mcp add opendecider -- opendecider mcp                 # register it with Claude Code

Tools: `decide` (any number of typed questions about one state), and the shortcuts `choose`, `yes_no` and `score` for a
single question (the shared core in `opendecider.tools`). Every answer has a probability for each option, so the calling
agent can act on confident answers and ask the user about the rest. The model loads on the first call, so the client's
handshake never waits for a download; inference runs one call at a time, off the event loop.

The server speaks MCP over stdio: stdout carries the protocol, and logs and download progress go to stderr.
"""
from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

from . import tools
from .tools import MAX_OPTIONS, MAX_QUESTIONS, MAX_STATE_CHARS, ModelError

__all__ = ["Decider", "build_server", "run", "INSTRUCTIONS",
           "MAX_OPTIONS", "MAX_QUESTIONS", "MAX_STATE_CHARS", "ModelError"]   # the limits lived here up to 0.3.0

log = logging.getLogger("opendecider.mcp")

INSTRUCTIONS = tools.INSTRUCTIONS


def _stdout_to_stderr():
    """Library output during load and inference goes to stderr: on stdio, stdout is the protocol. (The MCP SDK moves
    fd 1 to stderr, but Python's own stdout buffer would still be flushed onto the protocol at exit.)"""
    return contextlib.redirect_stdout(sys.stderr)


class Decider(tools.Decider):
    """`tools.Decider` with library output kept off the protocol stream."""

    def __init__(self, model: Any = tools.DEFAULT_MODEL, loader=None, **load_kw):
        super().__init__(model, loader, guard=_stdout_to_stderr, **load_kw)


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

    async def ask(fn, *args) -> dict:   # off the event loop: inference blocks
        try:
            return await anyio.to_thread.run_sync(fn, decider, *args)
        except (ValueError, ModelError) as e:   # the caller's input, or why the model is unavailable
            raise ToolError(str(e)) from None
        except Exception as e:   # noqa: BLE001 -- an inference failure: say what failed; the traceback goes to stderr
            log.exception("inference failed")
            raise ToolError(f"inference failed: {type(e).__name__}: {e}") from None

    @server.tool(annotations=pure, description=tools.DESCRIPTIONS["decide"])
    async def decide(state: str | dict | list, questions: dict[str, dict]) -> dict[str, Any]:
        return await ask(tools.decide, state, questions)

    @server.tool(annotations=pure, description=tools.DESCRIPTIONS["choose"])
    async def choose(state: str | dict | list, question: str, options: list[str] | dict[str, str]) -> dict[str, Any]:
        return await ask(tools.choose, state, question, options)

    @server.tool(annotations=pure, description=tools.DESCRIPTIONS["yes_no"])
    async def yes_no(state: str | dict | list, question: str) -> dict[str, Any]:
        return await ask(tools.yes_no, state, question)

    @server.tool(annotations=pure, description=tools.DESCRIPTIONS["score"])
    async def score(state: str | dict | list, question: str, levels: list[str]) -> dict[str, Any]:
        return await ask(tools.score, state, question, levels)

    return server


def run(model: str = tools.DEFAULT_MODEL, **load_kw) -> None:
    # stderr only (stdout carries the protocol); our own messages at INFO, other libraries' (HTTP requests) at WARNING
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("opendecider").setLevel(logging.INFO)
    build_server(Decider(model, **load_kw)).run("stdio")
