# Changelog

All notable changes to the `opendecider` package, and from 0.6.0 to `opendecider-client` (the same package without
PyTorch) and `@opendecider/client` on npm, which share its version number. Versions follow
[Semantic Versioning](https://semver.org). The public API is `load`, `OpenDecider.system_one`, `system_one_batch`, the
question helpers, the answer fields, the HTTP wire format and, from 0.4.0, the agent-facing surfaces: the MCP server's
tool names, arguments and answers, `opendecider.tools` and the public names in `opendecider.integrations`, from 0.5.0,
`opendecider.guard`, and from 0.6.0, the exports of `@opendecider/client` and its `/ai-sdk` and `/mastra` entry points
(where a deprecation is marked `@deprecated` and logs a warning).

- **Before 1.0.0** (0.x): patch releases (0.2.x) never change the public API. A breaking change can only ship in a new
  minor release (0.3.0, 0.4.0, …), and only after at least one release in which the old behaviour emits a deprecation
  warning.
- **From 1.0.0 on:** backward-compatible additions ship in minor releases, and breaking changes only in major releases,
  again with a deprecation release first.

## [Unreleased]

### Added

- `opendecider-client`: OpenDecider without PyTorch. The same package (`import opendecider`, routers, tools, the guard,
  the MCP server and every framework extra), built by `packaging/build_client.py` without torch, transformers and the
  local-model extras, for models served by `opendecider serve`, Ollama, LM Studio or vLLM. Install it or
  `opendecider`, not both. CI runs it in a clean environment against a live `opendecider serve`.
- `@opendecider/client` on npm: OpenDecider for TypeScript (tested on Node 22+, Bun and Deno), with no runtime
  dependencies. `load`, `systemOne` / `systemOneBatch`, `Router` and `Guard` against `opendecider serve`, or directly
  against Ollama, LM Studio and vLLM. `@opendecider/client/ai-sdk`: the MCP server's seven tools for the Vercel AI SDK
  (`ai` 5 to 7) and `guardMiddleware()` for `wrapLanguageModel`. `@opendecider/client/mastra`: the tools for Mastra
  (`@mastra/core` 1.11 or later) and `GuardProcessor`, an input processor. Its prompt, answers, limits and guard
  windows are tested against reference outputs from this package. Its version follows this package's.

### Changed

- A model that needs a missing package now says what to install, instead of a bare `No module named ...`: in
  `opendecider-client`, the full package (for a local model, `opendecider serve` or `bench-speed`); in `opendecider`,
  the extra (`pip install "opendecider[mlx]"`). Such a load failure is logged as one line, without a traceback.
- `import opendecider` warns when both `opendecider` and `opendecider-client` are installed: they share the same
  files, so uninstalling either one removes them for both.

## [0.5.0] - 2026-10-02

### Added

- `opendecider.guard`: `Guard` screens text for jailbreaks and prompt injection before an agent acts on it, with two
  yes/no checks (Laya's guard-preset attack questions) and per-model thresholds measured on public datasets.
  `check`, `check_many` (shared batches), `acheck`, `enforce` / `aenforce` (raise `GuardrailError`); a `GuardResult`
  for every check (passed, reason, violations, probabilities, thresholds, model, latency, windows); long text in
  overlapping windows; `on_error="block"` (default) or `"allow"`; `on_decision` hooks and an `opendecider.guard`
  OpenTelemetry span. Default model opendecider-small-td.
- The guard in each framework's hook: LangChain `guardrail_runnable()` (modes `raise`, `annotate`, `filter` for
  retrieved documents; batched), Agno `guardrail()` (Agno 2.1+), CrewAI `kickoff_guardrail()` and `task_guardrail()`,
  Google ADK `guardrail_callback()`, Microsoft Agent Framework `guardrail_middleware()`, PydanticAI
  `guardrail_capability()` (2.x) and `guardrail_processor()` (1.x), Strands `guardrail_hook()`, and a `guard` tool in
  `opendecider mcp`.
- LangChain `decision_runnable()`: typed answers about each input, with `batch` in batched model calls, and
  `decision_evaluator()`, an evaluator for LangSmith's `evaluate` that scores runs with one question instead of an LLM
  judge.
- LlamaIndex `DecisionMultiSelector`: picks every query engine likely to help, with one yes/no question per engine in a
  single forward pass.
- `benchmarks/guard.py`: the guard benchmark against Laya on deepset/prompt-injections,
  jackhhao/jailbreak-classification and xTRam1/safe-guard-prompt-injection, with the committed results.
- `examples/prompt_guard.py`.

### Changed

- Loading opendecider-small, -small-td, -medium-td or -large-td without `peft` now says to install
  `opendecider[small]`, instead of a bare `No module named 'peft'`.

### Security

- Releases are signed: the GitHub Release carries the wheel and sdist with their Sigstore signatures, GitHub build
  provenance and an SPDX SBOM, beside PyPI's own attestations. The Docker images on GHCR are signed with cosign
  (keyless) and carry an SBOM and build provenance in the registry. SECURITY.md has the commands to verify each.
- The Docker images build from base images pinned by digest and install every Python package by hash
  (`docker/lock.sh`); the CUDA image's copies of click, idna and pygments are updated past published advisories.
- CI: every GitHub Action pinned to a commit, read-only workflow tokens, OpenSSF Scorecard, a ruff lint job. The
  project holds the OpenSSF Best Practices passing badge.

## [0.4.0] - 2026-10-01

### Added

- `opendecider.integrations.langchain` (`pip install "opendecider[langchain]"`): `decision_tools()` gives LangChain
  agents the `decide`, `choose`, `yes_no` and `score` tools, and `DecisionRouter` is a LangGraph conditional edge that
  routes to a fallback node when the top route's probability is below `min_confidence`.
- `opendecider.integrations.llamaindex` (`pip install "opendecider[llamaindex]"`): `DecisionSelector` picks a
  RouterQueryEngine's query engine in one forward pass, in place of an LLM selector; `decision_tools()` gives the same
  four tools as `FunctionTool`s.
- Agno (`opendecider[agno]`): `decision_toolkit()`, a Toolkit with the four tools and instructions on using
  confidence, and `DecisionRouter.selector()`, the selector of a workflow `Router` step. Agno 2.x and 3.x.
- CrewAI (`opendecider[crewai]`): `decision_tools()` for crew agents, and `DecisionRouter`, whose result is the label a
  Flow's `@listen` methods wait for.
- Microsoft Agent Framework (`opendecider[agent-framework]`): `decision_tools()`, and `DecisionRouter.cases()`, the
  cases of a switch-case edge group (one forward pass per message, however many cases).
- Google ADK (`opendecider[google-adk]`): `decision_tools()`, and `DecisionRouterAgent`, which hands each request to
  one of its sub-agents.
- PydanticAI (`opendecider[pydantic-ai]`): `decision_toolset()`; invalid calls raise `ModelRetry` so the model fixes
  them.
- Strands Agents (`opendecider[strands]`): `decision_tools()` and `DecisionRouter`.
- Mastra and other TypeScript or MCP-capable frameworks use `opendecider mcp`; an example shows Mastra's `MCPClient`.
- Production routing, in every integration: the model can be an `opendecider serve` URL (`load("https://...")`,
  `model="https://..."`), so the routing process holds no model and answers are the server model's own;
  `router.decide()` returns a `tools.Decision` (route, reason, choice, confidence, probabilities, model, latency,
  error); `on_decision=` receives every decision, failed ones included; `on_error="fallback"` takes the fallback
  route when the decision fails; OpenTelemetry spans for every decision (`pip install "opendecider[otel]"`).
- CrewAI `TaskAssigner`: picks the crew member for each task from the members' roles and goals, in place of a
  hierarchical crew's manager LLM, with a fallback member for tasks no one fits confidently.
- MCP server: `decide_batch` (the same questions about up to 256 states in one call) and `status` (the model, whether
  it is loaded, the version and the limits, without loading it).
- `load(..., api_key=, timeout=)` for served models; a busy server's `Retry-After` is honoured.
- `opendecider.tools`: the shared core of the MCP server and every integration (validation, lazy loading, the
  agent-facing answer format, and `Router`, the routing logic every router builds on), so all of them answer alike.

### Changed

- A model that fails to load is retried after 5 seconds instead of on every call (`tools.LOAD_RETRY_S`), so a server
  outage costs one slow call rather than one per request; calls in between fail at once with the same reason. The
  same holds for a served model (`opendecider serve`, LM Studio, Ollama, vLLM) that times out or cannot be reached
  after loading (`remote.DOWN_RETRY_S`).
- Versioning: the agent-facing surfaces (the MCP server's tool names, arguments and answers, `opendecider.tools` and
  the public names in `opendecider.integrations`) are now part of the public API, under the same versioning policy.

### Documentation

- Examples: `examples/agent_frameworks/`, one per framework (LangGraph, LlamaIndex, Agno, CrewAI, Microsoft Agent
  Framework, Google ADK, PydanticAI, Strands, MCP, and Mastra in TypeScript), run in CI with opendecider-nano.
- [Agent frameworks](https://manjunathshiva.github.io/opendecider/guides/agent-frameworks/) guide covering each
  integration.
- Model servers: PyTorch is installed with the package but not used when LM Studio, Ollama or vLLM runs the model.

## [0.3.0] - 2026-10-01

### Added

- `opendecider mcp`: an MCP server, so AI assistants and agents (Claude Code, Claude Desktop, Cursor, …) can call
  OpenDecider as a tool (`pip install "opendecider[mcp]"`). Tools: `decide` (any number of typed questions) and the
  shortcuts `choose`, `yes_no` and `score`, all read-only, with a probability for every option. The model loads on
  the first call; invalid input is a tool error that names the problem; the same limits as `opendecider serve`.

### Fixed

- `opendecider serve`: an unexpected internal error now returns the JSON error body
  (`{"detail": "internal server error"}`) and the `x-request-id` header like every other response, instead of a
  plain-text 500 without the id; the traceback is logged with the request id.
- `opendecider serve`: an inference failure is now logged with the request id of the request that failed, as
  SECURITY.md describes; before, the log line did not say which request it was.
- `opendecider serve`: when a batch that merges several requests fails and each request is retried on its own, the
  failure is now logged as a warning; before, it was invisible when every retry succeeded.

### Security

- `opendecider serve`: a client's `x-request-id` is echoed and logged only if it is up to 128 letters, digits and
  `.` `_` `:` `-`; any other value is replaced by a generated id, so a client cannot write arbitrary text into the
  server log.

### Documentation

- Documentation site at https://manjunathshiva.github.io/opendecider/ (built with Zensical from `docs/`, published
  by `.github/workflows/docs.yml`): getting started, choosing a model, guides for serving, LM Studio / Ollama / vLLM,
  confident automation and agent guardrails, benchmarks, limitations, and Python, HTTP and command-line reference.
- New [examples](examples/): support triage, an agent guardrail, automating only the confident decisions (measured
  on the typed-decisions test split), an HTTP client for `opendecider serve`, and LM Studio / Ollama / vLLM. CI runs
  them against opendecider-nano on CPU whenever the package or the examples change (`remote_backends.py`, which needs
  a model server, is compile-checked).
- The Colab notebook covers small-td, batching, confident automation and `opendecider serve`, and is saved with the
  outputs of a run on an NVIDIA T4. It removes Colab's preinstalled torchao 0.10 first: recent peft refuses to load
  LoRA adapters next to it, so opendecider-small failed to load on Colab.
- Installation details (README): on Colab, `pip uninstall -y torchao` before loading small or small-td.

## [0.2.1] - 2026-09-30

### Added
- `load("lmstudio:<model>")`, `load("ollama:<model>")` and `load("openai:<model>", base_url=...)`: use a model served by
  LM Studio or Ollama (the GGUF builds), vLLM (the base model with the LoRA adapter), or another server with an
  OpenAI-compatible chat endpoint that returns token log-probabilities (`opendecider.remote`). LM Studio, Ollama and
  vLLM 0.30 are tested. OpenDecider sends the prompt the model was trained on and reads the option probabilities
  from `top_logprobs`. On typed-decisions (2,000 questions), the Q8_0 GGUF builds give the same top answer as the
  PyTorch model on 1,975 (small) and 1,972 (small-td), at the same accuracy (0.669 vs 0.671, 0.794 vs 0.792); the Q4_K_M
  builds agree on 1,857 and 1,875. Through vLLM (bf16, adapter not merged): 0.6735 and 0.7945 against 0.6715 and 0.792,
  the same top answer on 1,963 and 1,969.

  `opendecider serve --model lmstudio:...` puts the Jev-compatible `/v1/systemone` in front of it. Up to 26 options per
  question; the questions of a request run in parallel (4 at a time); transient upstream errors (429/5xx, reset
  connections) are retried twice; `opendecider serve`'s `/ready` reports 503 while the upstream server is unreachable.
  Credentials (`OPENDECIDER_REMOTE_API_KEY`) go only to the configured origin, never onto a redirect, and over plain
  HTTP only to this machine (use https for another host, or set `OPENDECIDER_REMOTE_ALLOW_HTTP=1`). Under
  `opendecider serve` the upstream timeout is capped at the server's request timeout.

### Changed
- The prompt the Qwen-based models were trained on now lives in the torch-free `opendecider.prompt`, so the LM Studio /
  Ollama backend does not import torch; `opendecider.small.render` still works.

## [0.2.0] - 2026-09-30

### Added
- `opendecider serve` (`pip install "opendecider[serve]"`): an HTTP server speaking TypeSafe Jev's `/v1/systemone`
  protocol. Verified with TypeSafe's own SDK (`@typesafe-ai/sdk` 0.6.0) by changing only the base URL.
  - `POST /v1/systemone`, `POST /v1/systemone/batch`, `GET /v1/models`, `GET /health`, `GET /ready`, `GET /metrics`
    (Prometheus).
  - Dynamic batching across concurrent requests.
  - Bounded admission: 503 with `Retry-After`, then a 504 request timeout.
  - Limits on body size, questions, options and state length, each answering a 4xx with a message.
  - Bearer auth (constant-time), request ids, configuration through `OPENDECIDER_*` environment variables.
- The `opendecider` command (`serve`, `bench-speed`).
- Docker images (CPU and CUDA) and compose files. Tagged releases publish `ghcr.io/manjunathshiva/opendecider`.
- `OpenDecider.system_one_batch(states, questions)`: many states in one call (one padded batch for nano).
- Responses carry `usage.input_tokens`. Answers are marked `truncated`, with a top-level `warnings` list, when a
  state was cut to fit the model's input length (nano: 2,048 tokens). Before this, truncation was silent.
- Score answers carry `legend` (level index to description), as Jev's do.
- `benchmarks/load_test.py`: latency percentiles, throughput and errors under a ramp of concurrent users, with charts.
  Results for CPU and NVIDIA L4 are in the README (100 users, 0 errors in every run).
- `--dtype bfloat16` for nano (`load(..., dtype=...)`): 2.8× server throughput on CPUs with bf16 units. On typed-decisions it
  has the same accuracy as fp32 (0.796) and the same top answer on 1,992 of 2,000 questions. The default stays float32.
- `--small-batch N` for Qwen-based models: several questions per forward pass, 2.2× throughput on an L4, and the top
  answer changes on about 1 question in 75. The default (1) keeps the exact evaluated path.
- Model revision pinning in the server (`--revision` / `OPENDECIDER_REVISION`).
- Security and maintenance: CodeQL (Python and Actions), Dependabot (pip, Docker base images, Actions), private
  vulnerability reporting, a protected `main` with required checks.

### Changed
- Development status: Production/Stable: the public API and the HTTP wire format follow the versioning policy at the
  top of this file (for 0.x, breaking changes only in a new minor release, after a deprecation release).
- Qwen-based models refuse an input longer than their context with a clear `ValueError` (HTTP 422) instead of
  failing inside torch.

## [0.1.2] - 2026-09-27
- `opendecider-medium-td` (Qwen3-30B-A3B) and multi-GPU loading (`device_map`). MLX builds can be a LoRA adapter on
  an `mlx-community` base.

## [0.1.1] - 2026-09-27
- MLX backend for Apple Silicon (`[mlx]` extra), `opendecider-small-td`, fp16 fallback on GPUs without bf16 (T4),
  a Colab notebook and the benchmark harness.

## [0.1.0] - 2026-09-26
- First release: `opendecider-nano` and `opendecider-small`.
