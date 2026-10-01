# Python API

The public API is `load`, `OpenDecider.system_one`, `OpenDecider.system_one_batch`, the question helpers `Choice`,
`Score` and `Noul`, and the answer fields below. Patch releases never change it; see the
[versioning policy](https://github.com/manjunathshiva/opendecider/blob/main/CHANGELOG.md).

## `load`

```python
load(name_or_path="manjunathshiva/opendecider-nano", device=None, revision=None, dtype=None, base_url=None) -> OpenDecider
```

| argument | meaning |
|---|---|
| `name_or_path` | a Hugging Face model name, a local folder with `opendecider.json`, or `lmstudio:<model>`, `ollama:<model>`, `openai:<model>` for a model served by an app ([LM Studio, Ollama and vLLM](../guides/model-servers.md)) |
| `device` | `"cuda"`, `"mps"` or `"cpu"`; default: CUDA, then MPS, then CPU. Ignored for the MLX builds and served models |
| `revision` | a Hub revision (tag, branch or commit) to pin |
| `dtype` | nano only: `"float32"` (default, as evaluated) or `"bfloat16"` (faster on CPUs with bf16 units and on GPUs) |
| `base_url` | the server URL for `openai:` models (or set `OPENDECIDER_REMOTE_URL`) |

Which package extra a model needs: nano none; small, small-td, medium-td and large-td `opendecider[small]`; the MLX
builds `opendecider[mlx]`; served models none.

## `OpenDecider.system_one`

```python
model.system_one(state, questions) -> dict
```

- `state`: a string, or any JSON-serialisable object (serialised to JSON text before the model reads it).
- `questions`: a non-empty dict of named questions, each a dict or a `Choice` / `Score` / `Noul`.

Returns:

```python
{"model": "opendecider-nano",
 "answers": {"department": {...}, "urgency": {...}},
 "usage": {"input_tokens": 106, "output_tokens": 0},
 "latency_ms": 17.4}
```

plus `"warnings": [...]` when a state was truncated to fit the model's input length. Invalid questions raise
`ValueError` with a message naming the question.

## `OpenDecider.system_one_batch`

```python
model.system_one_batch(states, questions) -> list[dict]
```

The same questions about many states, one result per state (same shape as `system_one`, without `latency_ms`).
opendecider-nano runs them as one padded batch.

## Questions

```python
Choice(instructions, criteria)        # criteria: {"label": "description" or None, ...} or ["label", ...]; at least 2
Score(instructions, criteria)         # criteria: ordered levels, lowest first; at least 2
Noul(instructions, criteria={})       # yes / no; optional {"true": "...", "false": "..."} descriptions
```

As dicts: `{"type": "choice" | "score" | "noul", "instructions": "...", "criteria": ...}`.

## Answers

| field | choice | score | noul |
|---|---|---|---|
| `type` | `"choice"` | `"score"` | `"noul"` |
| the answer | `choice`: the most likely label | `score`: the most likely level (0 = lowest); `expected`: the expected level | `noul`: the probability of yes |
| `probabilities` | per label | per level, keyed `"0"`, `"1"`, … | `{"true": p, "false": 1 - p}` |
| `confidence` | probability of the top label | probability of the top level | `max(p, 1 - p)` |
| other | | `legend`: `{"0": "lowest level", …}` | |

An answer also carries `"truncated": true` when its state was cut to fit the model's input length.

!!! note "Score on the wire"
    Over HTTP (`opendecider serve`) a score answer follows TypeSafe Jev: `score` is the expected score and `level` the
    most likely level. In Python, `score` is the most likely level and `expected` the expected score.

## Agent tools and integrations

`opendecider.tools` is the core behind the [MCP server](../guides/mcp.md) and the
[LangChain, LangGraph and LlamaIndex](../guides/agent-frameworks.md) integrations:

```python
from opendecider.tools import Decider, choose, decide, score, yes_no

d = Decider("manjunathshiva/opendecider-nano")   # or tools.shared(name): one copy per model name
choose(d, state, "Which team?", {"billing": "charges", "tech": "bugs"})
# {'choice': 'billing', 'probabilities': {...}, 'confidence': ...}
yes_no(d, state, "Is this spam?")   # {'answer': 'yes' or 'no', 'probability_yes': ..., 'confidence': ...}
score(d, state, "How urgent?", ["low", "medium", "high"])   # {'level', 'label', 'expected_level', ...}
```

| integration | install | provides |
|---|---|---|
| `opendecider.integrations.langchain` | `opendecider[langchain]` | `decision_tools()`, `DecisionRouter` |
| `opendecider.integrations.llamaindex` | `opendecider[llamaindex]` | `decision_tools()`, `DecisionSelector` |

Invalid input raises `ValueError` and a model that cannot load raises `tools.ModelError`, each naming the problem.

## Environment variables for served models

| variable | meaning |
|---|---|
| `OPENDECIDER_REMOTE_URL` | server URL for `lmstudio:`, `ollama:` and `openai:` models when `base_url` is not given |
| `OPENDECIDER_REMOTE_API_KEY` | bearer token for that server; sent only to it, never on a redirect |
| `OPENDECIDER_REMOTE_ALLOW_HTTP` | `1` to allow sending the key over plain HTTP to another host on a trusted network |
