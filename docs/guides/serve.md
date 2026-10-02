# Serve it: a drop-in for TypeSafe Jev's API

`opendecider serve` puts any OpenDecider model behind an HTTP server that speaks TypeSafe Jev's `/v1/systemone`
protocol, so existing Jev clients work by changing the base URL. It was tested with TypeSafe's own SDK
(`@typesafe-ai/sdk` 0.6.0) and no code change.

## Start it

=== "pip"

    ```bash
    pip install "opendecider[serve]"
    opendecider serve --model manjunathshiva/opendecider-nano       # http://0.0.0.0:8000
    ```

=== "Docker (CPU)"

    ```bash
    docker run -p 8000:8000 -v opendecider-models:/models ghcr.io/manjunathshiva/opendecider:latest
    ```

=== "Docker (NVIDIA)"

    ```bash
    docker run --gpus all -p 8000:8000 -v opendecider-models:/models ghcr.io/manjunathshiva/opendecider:latest-cuda
    ```

=== "Docker Compose"

    ```bash
    git clone https://github.com/manjunathshiva/opendecider && cd opendecider
    docker compose up                                              # CPU
    docker compose -f compose.yaml -f compose.cuda.yaml up         # NVIDIA
    ```

The images serve opendecider-nano by default; set `OPENDECIDER_MODEL` to serve another model, and mount a volume at
`/models` so downloaded weights survive restarts. Pin a release with a version tag, for example
`ghcr.io/manjunathshiva/opendecider:0.4.0`. From 0.6.0 the CPU image runs natively on `linux/amd64` and `linux/arm64`
(Apple Silicon, AWS Graviton, Ampere), and Docker pulls the one that fits; before 0.6.0 it was `linux/amd64` only, so an
Apple Silicon Mac ran it emulated and many times slower. The NVIDIA image is `linux/amd64`.

## Call it

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Hi, we were billed twice for March. Please refund the duplicate today.",
  "questions": {"department": {"type": "choice", "instructions": "Which department?",
                               "criteria": {"billing": "payments, refunds", "technical": "bugs, outages"}}}}'
```

With TypeSafe's SDK, point it at the server:

```bash
TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=<your OPENDECIDER_API_KEY> node app.js
```

A Python client with retries (standard library only) is in
[examples/serve_client.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/serve_client.py).

From Python, the MCP server and every agent framework integration, pass the server's URL as the model
(`load("http://localhost:8000")`, `DecisionRouter(..., model="http://localhost:8000")`): its model answers, with the
same answers as in-process, and the client loads none. Set `OPENDECIDER_REMOTE_API_KEY` to the server's key. See
[Production](agent-frameworks.md#production).

### Without PyTorch: opendecider-client

A process that only calls the server does not need PyTorch (about 2 GB) or transformers. `opendecider-client` is the
same package, `import opendecider` included, built without them:

```bash
pip install opendecider-client                 # or "opendecider-client[langchain]", [mcp], ... as for opendecider
```

```python
from opendecider import load
from opendecider.guard import Guard
from opendecider.integrations.langchain import DecisionRouter

model = load("http://localhost:8000")          # every router, tool and guard takes the URL as model=
route = DecisionRouter({"billing": "charges, refunds", "tech": "bugs"}, "Which team?", model="http://localhost:8000")
guard = Guard(model="http://localhost:8000")   # the measured threshold of the served model applies
```

It cannot run a model on this machine: `load("manjunathshiva/opendecider-nano")` and `opendecider serve` say to
install `opendecider` instead. Install one of the two packages, not both: they share the `opendecider` files (like
opencv-python and opencv-python-headless), so uninstalling one removes them for the other. `import opendecider` warns
when both are installed; to keep one, `pip uninstall -y opendecider-client && pip install --force-reinstall opendecider`
(or the reverse). Ollama, LM Studio and vLLM models work too (`ollama:...`).

| endpoint | what it does |
|---|---|
| `POST /v1/systemone` | one state, any number of typed questions (Jev's request and response shape) |
| `POST /v1/systemone/batch` | `{"states": [...], "questions": {...}}`: the same questions about many states in one call |
| `GET /v1/models` | the served model (as Jev's models list) |
| `GET /health`, `GET /ready` | liveness and readiness probes (model, device, version, queue depth) |
| `GET /metrics` | Prometheus text: requests by status, latency and batch-size histograms, in-flight, queue depth |

Request and response fields, limits and status codes: [HTTP API](../reference/http-api.md). Every flag and environment
variable: [Command line](../reference/cli.md).

## Production behaviour

- **Batching across requests:** one inference thread merges the questions of all waiting requests into shared batches,
  so nano's batched speed holds under concurrent load.
- **Bounded load:** above `--max-in-flight` requests the server answers 503 with `Retry-After` at once instead of
  queueing without limit, and a request not answered within `--request-timeout-s` gets 504.
- **Validation:** request size, questions per state, options per question and state length are all bounded, and invalid
  requests get a 4xx with a message naming the problem.
- **Security:** bearer-token auth with `OPENDECIDER_API_KEY` (constant-time comparison); server errors never leak
  internals to the client.
- **Observability:** an `x-request-id` header on every response, and Prometheus metrics at `/metrics`.

## Performance under load

`benchmarks/load_test.py`: 100 concurrent users, 20 s ramp, 40 s hold, three questions per request, one server process
(raw results in [benchmarks/results/load/](https://github.com/manjunathshiva/opendecider/tree/main/benchmarks/results/load)):

| model and hardware | requests / s | p50 | p95 | p99 | errors |
|---|---|---|---|---|---|
| nano, CPU only (8 cores, AWS c7i.4xlarge), `--dtype bfloat16` | **24** | 4.4 s | 4.9 s | 5.2 s | 0 |
| nano, CPU only (same machine), default fp32 | 9 | 16.6 s | 17.2 s | 19.0 s | 0 |
| nano, 1× NVIDIA L4 | **50** | 2.1 s | 2.3 s | 2.3 s | 0 |
| small (4B), 1× NVIDIA L4, `--small-batch 16` | **20** | 5.8 s | 6.5 s | 6.6 s | 0 |
| small (4B), 1× NVIDIA L4, default | 9 | 14.4 s | 14.7 s | 16.1 s | 0 |

![opendecider serve: latency percentiles and throughput at 100 concurrent users on CPU and on an NVIDIA L4](https://raw.githubusercontent.com/manjunathshiva/opendecider/main/assets/load_test.png)

With 100 users each waiting for their answer before sending the next, latency is roughly users ÷ throughput, so a single
process is at its limit here. To serve more traffic, run more replicas behind a load balancer (one process per GPU; on
CPU, one process per 8 or so physical cores).

## Recommended settings

- **CPU with bf16 units** (Intel Sapphire Rapids and newer, i.e. AMX): `--dtype bfloat16`, 2.8× the throughput. On
  typed-decisions it has the same accuracy as fp32 (0.796) and the same top answer on 1,992 of 2,000 questions. On CPUs
  without bf16 units keep the default.
- **Qwen-based models on a GPU** (small, small-td, medium-td, large-td): `--small-batch 16`, 2.2× the throughput. The
  top answer changes on about 1 question in 75 (bf16 arithmetic in padded batches), so leave it off where you need
  exactly the published answers.
- `--threads` at most the number of physical cores; oversubscribing hyperthreads slows CPU inference.

## Serve a model running in LM Studio, Ollama or vLLM

`--model` also takes a model served elsewhere; the server then answers `/v1/systemone` using that app as the engine:

```bash
opendecider serve --model lmstudio:opendecider-small
opendecider serve --model ollama:hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0
OPENDECIDER_REMOTE_URL=http://localhost:8001/v1 opendecider serve --model openai:opendecider-small   # e.g. vLLM
```

The last line assumes vLLM was started with `--port 8001`, since `opendecider serve` itself listens on 8000.

See [LM Studio, Ollama and vLLM](model-servers.md).
