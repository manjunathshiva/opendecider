# Command line

## `opendecider serve`

```bash
pip install "opendecider[serve]"
opendecider serve [--model NAME] [flags]
```

Every flag can also be set as an environment variable, `OPENDECIDER_<NAME>`; flags win.

| flag | environment variable | default | meaning |
|---|---|---|---|
| `--model` | `OPENDECIDER_MODEL` | `manjunathshiva/opendecider-nano` | Hub name, local folder, or `lmstudio:` / `ollama:` / `openai:` model |
| `--revision` | `OPENDECIDER_REVISION` | latest | pin a Hub revision (tag, branch or commit) |
| `--device` | `OPENDECIDER_DEVICE` | auto | `cuda`, `mps` or `cpu` |
| `--host` | `OPENDECIDER_HOST` | `0.0.0.0` | |
| `--port` | `OPENDECIDER_PORT` | `8000` | |
| `--max-batch` | `OPENDECIDER_MAX_BATCH` | `32` | questions per inference batch |
| `--batch-wait-ms` | `OPENDECIDER_BATCH_WAIT_MS` | `2` | how long to wait to fill a batch |
| `--max-in-flight` | `OPENDECIDER_MAX_IN_FLIGHT` | `256` | admitted requests before answering 503 |
| `--request-timeout-s` | `OPENDECIDER_REQUEST_TIMEOUT_S` | `30` | 504 after this long |
| `--threads` | `OPENDECIDER_THREADS` | torch's choice | torch CPU threads; at most the number of physical cores |
| `--dtype` | `OPENDECIDER_DTYPE` | `float32` | nano: `bfloat16` is faster on CPUs with bf16 units |
| `--small-batch` | `OPENDECIDER_SMALL_BATCH` | `1` | Qwen-based models: questions per forward pass (1 = the exact published path) |
| `--log-level` | `OPENDECIDER_LOG_LEVEL` | `info` | |

Settings with no flag (environment only):

| environment variable | default | meaning |
|---|---|---|
| `OPENDECIDER_API_KEY` | none | require `Authorization: Bearer <key>` |
| `OPENDECIDER_MAX_BODY_BYTES` | `1048576` | larger request bodies get 413 |
| `OPENDECIDER_MAX_QUESTIONS` | `64` | per state |
| `OPENDECIDER_MAX_OPTIONS` | `256` | per question |
| `OPENDECIDER_MAX_STATE_CHARS` | `200000` | JSON-serialised state |
| `OPENDECIDER_MAX_BATCH_STATES` | `256` | states per `/v1/systemone/batch` request |
| `OPENDECIDER_ROOT_PATH` | none | public URL prefix behind a reverse proxy |
| `OPENDECIDER_REMOTE_URL` | none | server URL for `openai:` models |

## `opendecider mcp`

```bash
pip install "opendecider[mcp]>=0.5.0"
opendecider mcp [--model NAME] [--revision REV] [--device DEV] [--dtype DTYPE]
```

An MCP server over stdio for AI assistants and agents; see [AI assistants (MCP)](../guides/mcp.md). `--model` defaults to
`OPENDECIDER_MODEL`, else opendecider-nano; the other flags work as for `opendecider serve`.

## `opendecider bench-speed`

```bash
opendecider bench-speed manjunathshiva/opendecider-nano
```

Latency at 1, 5, 10 and 50 questions per call on this machine (also `python -m opendecider.bench_speed <model>`).
