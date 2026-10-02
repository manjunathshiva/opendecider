# AGENTS.md

Context for AI coding assistants (Claude Code, Codex, Cursor, Copilot, Gemini CLI) working in this repository. The
rules mirror [CONTRIBUTING.md](CONTRIBUTING.md); coding agents read this file automatically.

**OpenDecider** is a family of open decision models and the `opendecider` Python package: typed questions (`choice`,
`score`, `noul`) about a state, answered with a calibrated probability for every option, plus `opendecider serve`, an
HTTP server compatible with TypeSafe Jev's `/v1/systemone` protocol.

## Do NOT

- **Change published answers silently.** Anything that changes how a model reads or scores inputs
  (`opendecider/prompt.py`, `nano.py`, `small.py`, `mlx_small.py`, `questions.py`) changes the published benchmark
  numbers. Such a change needs before/after results from `benchmarks/` in the pull request.
- **Break the wire format.** `opendecider/serve.py` must keep answering Jev's request and response shape (a score
  answer's `score` is the expected score and `level` the most likely level). Keep auth comparisons constant-time, give
  every limit a 4xx with a message, and never return tracebacks or paths to the client.
- **Change the public API** (`load`, `system_one`, `system_one_batch`, `Choice` / `Score` / `Noul`, the answer fields,
  the MCP tools, `opendecider.tools`, `opendecider.guard`, the public names in `opendecider.integrations`, and the
  exports of `@opendecider/client`) outside the versioning policy in [CHANGELOG.md](CHANGELOG.md).
- Add a dependency to the core package, or one that needs a hosted service. Optional features go in an extra in
  `pyproject.toml` (`small`, `mlx`, `serve`). `@opendecider/client` has no runtime dependencies; frameworks are optional
  peer dependencies.
- Reformat files wholesale, reorder imports or "modernise" surrounding code. Match the style of the file being edited
  (lines up to 120 characters).
- Commit secrets, tokens, model weights or other large binaries.

## Where to look

| editing | check |
|---|---|
| `opendecider/` (any) | `python -m pytest -q tests` (torch-free; a fake model stands in, no download) |
| `opendecider/serve.py` | `tests/test_serve.py` |
| `opendecider/remote.py` (LM Studio / Ollama / vLLM backend; `opendecider serve` by URL) | `tests/test_remote.py`, `tests/test_served.py` |
| `opendecider/questions.py` | `tests/test_questions.py` |
| `opendecider/tools.py`, `mcp_server.py` | `tests/test_mcp.py`, `tests/test_decisions.py` (Decision, hooks, `on_error`, spans) |
| `opendecider/guard.py` | `tests/test_guard.py`; its framework hooks: `tests/test_guard_frameworks.py` (CrewAI and Strands in their own venv, as below); accuracy: `python benchmarks/guard.py report` |
| `opendecider/integrations/` (LangChain, LangGraph, LlamaIndex) | `tests/test_integrations.py` |
| `opendecider/integrations/` (Agno, CrewAI, Agent Framework, Google ADK, PydanticAI, Strands) | `tests/test_frameworks.py`; CrewAI and Strands pin `mcp` 1.x, so run them in a separate venv (`pip install crewai strands-agents`) |
| `typescript/` (`@opendecider/client` on npm) | `cd typescript && npm ci --ignore-scripts && npm run typecheck && npm test` (fake servers, no network), and `npm run format` (Prettier, 120 columns); behaviour shared with Python must match `test/fixtures/parity.json`: after changing the prompt, questions, tools or guard in either language, run `python typescript/test/fixtures/make_fixtures.py`; live: `OPENDECIDER_LIVE_URL=http://127.0.0.1:8000 npx vitest run test/live.test.ts` |
| `packaging/` (opendecider-client) | `tests/test_packaging.py`; CI's examples job installs the built client in a clean environment and runs `packaging/check_client.py` against `opendecider serve` |
| `examples/` | run the script with opendecider-nano on CPU; CI runs them all |
| `docs/`, `zensical.toml` | `pip install -r requirements-docs.txt && zensical build --strict --clean` |
| `benchmarks/` | `python benchmarks/report.py` rebuilds every table from the committed results |

Development setup:

```bash
pip install -e ".[small,serve,dev]"
python -m pytest -q tests
```

## Commits and pull requests

- Short imperative subjects with a scope prefix, as in the history: `docs: …`, `examples: …`, `serve: …`,
  `chore: …`. One logical change per commit.
- One focused change per pull request, rebased on the latest `main`.
- If a change can move numbers (accuracy, probabilities, latency, memory), report the before and after.
- Update `docs/`, `examples/` or `CHANGELOG.md` when behaviour or the public API changes.
- Write in English.
