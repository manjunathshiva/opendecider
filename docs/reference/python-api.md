# Python API

The public API is `load`, `OpenDecider.system_one`, `OpenDecider.system_one_batch`, the question helpers `Choice`,
`Score` and `Noul`, the answer fields below and, from 0.4.0, the agent-facing surfaces: the MCP server's tool names,
arguments and answers, `opendecider.tools` and the public names in `opendecider.integrations`. Patch releases never
change it; see the
[versioning policy](https://github.com/manjunathshiva/opendecider/blob/main/CHANGELOG.md).

## `load`

```python
load(name_or_path="manjunathshiva/opendecider-nano", device=None, revision=None, dtype=None, base_url=None,
     api_key=None, timeout=None) -> OpenDecider
```

| argument | meaning |
|---|---|
| `name_or_path` | a Hugging Face model name, a local folder with `opendecider.json`, `lmstudio:<model>`, `ollama:<model>`, `openai:<model>` for a model served by an app ([LM Studio, Ollama and vLLM](../guides/model-servers.md)), or the http(s) URL of an [`opendecider serve`](../guides/serve.md) (its model answers; nothing loads locally) |
| `device` | `"cuda"`, `"mps"` or `"cpu"`; default: CUDA, then MPS, then CPU. Ignored for the MLX builds and served models |
| `revision` | a Hub revision (tag, branch or commit) to pin |
| `dtype` | nano only: `"float32"` (default, as evaluated) or `"bfloat16"` (faster on CPUs with bf16 units and on GPUs) |
| `base_url` | the server URL for `openai:` models (or set `OPENDECIDER_REMOTE_URL`) |
| `api_key` | served models: the server's bearer token (default: `OPENDECIDER_REMOTE_API_KEY`) |
| `timeout` | served models: seconds to wait for each request (default 30 for `opendecider serve`, 120 for the apps) |

Which package extra a model needs: nano none; small, small-td, medium-td and large-td `opendecider[small]`; the MLX
builds `opendecider[mlx]`; served models none. A missing one raises `ImportError` with the install line. With
[`opendecider-client`](../guides/serve.md#without-pytorch-opendecider-client) (no PyTorch), only served models load:
a Hub name or local folder raises `ImportError` saying to install `opendecider` instead.

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
[agent framework](../guides/agent-frameworks.md) integrations:

```python
from opendecider.tools import Decider, choose, decide, score, yes_no

d = Decider("manjunathshiva/opendecider-nano")   # or tools.shared(name): one copy per model name
choose(d, state, "Which team?", {"billing": "charges", "tech": "bugs"})
# {'choice': 'billing', 'probabilities': {...}, 'confidence': ...}
yes_no(d, state, "Is this spam?")   # {'answer': 'yes' or 'no', 'probability_yes': ..., 'confidence': ...}
score(d, state, "How urgent?", ["low", "medium", "high"])   # {'level', 'label', 'expected_level', ...}
```

The integrations need opendecider 0.4.0 or later, and 0.5.0 for the guard hooks, `decision_runnable()`,
`decision_evaluator()` and `DecisionMultiSelector` (see [Agent frameworks](../guides/agent-frameworks.md)).

| integration | install | provides |
|---|---|---|
| `opendecider.integrations.langchain` | `opendecider[langchain]` | `decision_tools()`, `DecisionRouter`, `decision_runnable()`, `decision_evaluator()`, `guardrail_runnable()` |
| `opendecider.integrations.llamaindex` | `opendecider[llamaindex]` | `decision_tools()`, `DecisionSelector`, `DecisionMultiSelector` |
| `opendecider.integrations.agno` | `opendecider[agno]` | `decision_toolkit()`, `DecisionRouter` (`.selector()`), `guardrail()` |
| `opendecider.integrations.crewai` | `opendecider[crewai]` | `decision_tools()`, `DecisionRouter`, `TaskAssigner`, `kickoff_guardrail()`, `task_guardrail()` |
| `opendecider.integrations.agent_framework` | `opendecider[agent-framework]` | `decision_tools()`, `DecisionRouter` (`.cases()`), `guardrail_middleware()` |
| `opendecider.integrations.google_adk` | `opendecider[google-adk]` | `decision_tools()`, `DecisionRouterAgent`, `guardrail_callback()` |
| `opendecider.integrations.pydantic_ai` | `opendecider[pydantic-ai]` | `decision_toolset()`, `DecisionRouter`, `guardrail_capability()`, `guardrail_processor()` |
| `opendecider.integrations.strands` | `opendecider[strands]` | `decision_tools()`, `DecisionRouter`, `guardrail_hook()` |

`tools.Router(routes, instructions, *, model=..., fallback=None, min_confidence=0.0, on_error="raise",
on_decision=None)` is the router every integration builds on. Calling it with a state returns a route name (the
fallback when the top route's probability is below `min_confidence`); `.decide(state)` returns the full
`tools.Decision` (route, reason, choice, confidence, probabilities, model, latency, truncation, error), which
`on_decision` also receives and `.last` keeps. An empty state takes the fallback without a decision (or raises
`ValueError` without one). `on_error="fallback"` takes the fallback when the decision fails. See
[Production](../guides/agent-frameworks.md#production).

`tools.decide_batch(decider, states, questions)` answers the same questions about up to 256 states in one batch
(1,024 questions in all, states times questions);
`tools.status(decider)` reports the model, whether it is loaded, the version and the limits, without loading it.

Invalid input raises `ValueError`, a model that cannot load raises `tools.ModelError`, and a model server that cannot
answer raises `remote.ServerError`, each naming the problem. `tools.as_result(fn, *args)` returns those as
`{"error": "..."}` instead, for frameworks that hide a tool's exception text from the model.

## Prompt guard

```python
from opendecider.guard import Guard, GuardrailError

guard = Guard(checks=None, model="manjunathshiva/opendecider-small-td", threshold=None, on_error="block",
              on_decision=None, window_chars=4000)
guard.check(text)          # GuardResult (also guard(text)); guard.acheck(text) in async code
guard.check_many(texts)    # one GuardResult per text, short texts in shared batches
guard.enforce(text)        # the GuardResult, or raises GuardrailError (a ValueError) when the text is blocked
```

- `checks`: `{"name": "question"}`, yes/no questions that call the text `` `prompt` ``; the default is
  `opendecider.guard.ATTACK_CHECKS` (`jailbreak`, `prompt_injection`).
- `threshold`: a probability for every check, or `{"check": probability}`; by default the model's measured threshold
  for the default checks (`opendecider.guard.THRESHOLDS`: per model, and per published GGUF and MLX build), else 0.5.
  A check is flagged at or above its threshold.
- `on_error`: `"block"` blocks text that could not be checked, `"allow"` lets it through; either way `reason="error"`.
- `on_decision`: a function or list, called with every `GuardResult`; a hook that raises is logged and ignored.
- `GuardResult`: `passed`, `reason` (`passed`, `flagged`, `empty_input`, `error`), `violations`, `probabilities`,
  `thresholds`, `model`, `latency_ms`, `windows`, `truncated`, `error`, and `to_dict()`.

The framework hooks take a `Guard` or its settings as keywords; see
[Agent guardrails](../guides/agent-guardrails.md#in-your-agent-framework).

## Environment variables for served models

| variable | meaning |
|---|---|
| `OPENDECIDER_REMOTE_URL` | server URL for `lmstudio:`, `ollama:` and `openai:` models when `base_url` is not given |
| `OPENDECIDER_REMOTE_API_KEY` | bearer token for that server (and for an `opendecider serve` URL); sent only to it, never on a redirect |
| `OPENDECIDER_REMOTE_ALLOW_HTTP` | `1` to allow sending the key over plain HTTP to another host on a trusted network |
