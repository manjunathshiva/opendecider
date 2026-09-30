# Changelog

All notable changes to the `opendecider` package. Versions follow [Semantic Versioning](https://semver.org): from
0.2.0 on, the public API (`load`, `OpenDecider.system_one`, `system_one_batch`, the question helpers, the answer
fields and the HTTP wire format) changes only in a new minor version, with a deprecation first.

## [0.2.0] - unreleased

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

### Changed
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
