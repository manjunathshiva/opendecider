# Changelog

All notable changes to the `opendecider` package. Versions follow [Semantic Versioning](https://semver.org). The public
API is `load`, `OpenDecider.system_one`, `system_one_batch`, the question helpers, the answer fields and the HTTP wire
format.

- **Before 1.0.0** (0.x): patch releases (0.2.x) never change the public API. A breaking change can only ship in a new
  minor release (0.3.0, 0.4.0, …), and only after at least one release in which the old behaviour emits a deprecation
  warning.
- **From 1.0.0 on:** backward-compatible additions ship in minor releases, and breaking changes only in major releases,
  again with a deprecation release first.

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
