# HTTP API

`opendecider serve` speaks TypeSafe Jev's `/v1/systemone` protocol. A client written for Jev keeps working when pointed
at it; OpenDecider adds fields a Jev client ignores (`probabilities` and `confidence` on every answer, `warnings` when
an input was truncated).

## `POST /v1/systemone`

Request:

```json
{
  "model": "optional, ignored: the server answers with the model it serves",
  "state": "Hi, we were billed twice for March. Please refund the duplicate today.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which department?",
                   "criteria": {"billing": "payments, refunds", "technical": "bugs, outages"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "blocking"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}
  }
}
```

`state` is a string or any JSON value. Response:

```json
{
  "model": "opendecider-nano",
  "answers": {
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.95, "technical": 0.05}, "confidence": 0.95},
    "urgency": {"type": "score", "score": 1.4, "level": 2,
                "legend": {"0": "not urgent", "1": "soon", "2": "blocking"},
                "probabilities": {"0": 0.12, "1": 0.36, "2": 0.52}, "confidence": 0.52},
    "refund": {"type": "noul", "noul": 0.97, "probabilities": {"true": 0.97, "false": 0.03}, "confidence": 0.97}
  },
  "usage": {"input_tokens": 212, "output_tokens": 0}
}
```

(Illustrative numbers.) A score answer's `score` is the **expected** score, which can fall between levels, and `level`
is the most likely level, as in Jev. When a state was cut to fit the model, its answers carry `"truncated": true` and
the response has a `warnings` list.

## `POST /v1/systemone/batch`

```json
{"states": ["first state", {"any": "json"}], "questions": {...}}
```

Response: `{"model": ..., "results": [<one /v1/systemone response per state>], "total_usage": {...}}`.

## Other endpoints

| endpoint | response |
|---|---|
| `GET /v1/models` | `{"models": [{"name", "description", "release_date", "kind"}]}` |
| `GET /health` | `{"status": "ok", "model", "kind", "device", "version", "in_flight", "queue_depth"}` |
| `GET /ready` | `{"ready": true}`, plus `"upstream"` for a model served by LM Studio, Ollama or vLLM; 503 if the inference thread has stopped or that server is unreachable |
| `GET /metrics` | Prometheus text: `opendecider_requests_total`, `opendecider_request_seconds`, `opendecider_batch_questions`, `opendecider_questions_total`, `opendecider_in_flight`, `opendecider_queue_depth` |

## Authentication

Start the server with `OPENDECIDER_API_KEY` set and `POST /v1/systemone` and `POST /v1/systemone/batch` need
`Authorization: Bearer <key>` (compared in constant time); missing or wrong keys get 401. `GET /v1/models` and the
probes (`/health`, `/ready`, `/metrics`) stay open, so keep the server on a private network or put the probes behind
your proxy's rules.

## Limits and status codes

| status | when | default limit (setting) |
|---|---|---|
| 400 | the body is not valid JSON, or not a JSON object | |
| 401 | missing or wrong bearer token | |
| 413 | the body is too large | 1 MiB (`OPENDECIDER_MAX_BODY_BYTES`) |
| 422 | invalid questions, too many questions, options or states, or the state is too long | 64 questions per state, 256 options per question (26 for a model served by LM Studio, Ollama or vLLM), 200,000 characters of state, 256 states per batch |
| 503 | the server is at capacity (with `Retry-After: 1`) | 256 admitted requests (`--max-in-flight`) |
| 504 | no answer in time | 30 s (`--request-timeout-s`) |
| 500 | inference failed (no internals are returned) | |

Every error body is `{"detail": "<a message naming the problem>"}` (an unexpected internal error is
`{"detail": "internal server error"}`, with the traceback and the request id only in the server log), and every
response has an `x-request-id` header (the client's, if it sent one). A client should retry 503 after `Retry-After`; see
[examples/serve_client.py](https://github.com/manjunathshiva/opendecider/blob/main/examples/serve_client.py).
