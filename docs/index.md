# OpenDecider

<p align="center">
  <img src="https://raw.githubusercontent.com/manjunathshiva/opendecider/main/assets/logo-lockup.png#only-light" alt="OpenDecider" width="360" />
  <img src="https://raw.githubusercontent.com/manjunathshiva/opendecider/main/assets/logo-lockup-dark.png#only-dark" alt="OpenDecider" width="360" />
</p>

**Open, calibrated System 1 decision models.** Ask typed questions (`choice`, `score`, `noul`) about any state (text,
an email, a ticket or JSON) and get a calibrated probability for every option: 17 ms per question on an NVIDIA GPU,
18 ms on a Mac. Nothing is generated, so there is nothing to parse and nothing to hallucinate. Apache-2.0 code and
weights.

```bash
pip install opendecider
```

```python
from opendecider import Choice, Noul, Score, load

model = load("manjunathshiva/opendecider-nano")   # 0.8 GB, downloaded on first use

r = model.system_one(
    "We were billed twice for March. Refund the duplicate today or we cancel our plan.",
    {"department": Choice("Which department should handle this?",
                          {"billing": "invoices, payments, refunds",
                           "technical": "bugs, outages, system errors",
                           "other": "everything else"}),
     "urgency": Score("How urgent is this?", ["not urgent", "soon", "blocking"]),
     "churn_risk": Noul("Does the user threaten to cancel or leave?")})

r["answers"]["department"]["choice"]   # "billing", with a probability for every option
r["answers"]["urgency"]["score"]       # 2 = blocking
r["answers"]["churn_risk"]["noul"]     # probability the answer is yes
```

[Get started](getting-started.md){ .md-button .md-button--primary }
[Try the live demo](https://huggingface.co/spaces/manjunathshiva/opendecider-demo){ .md-button }
[Open in Colab](https://colab.research.google.com/github/manjunathshiva/opendecider/blob/main/notebooks/opendecider_colab.ipynb){ .md-button }

## Find a guide

| I want to… | Read |
|---|---|
| install it and make my first decision | [Get started](getting-started.md) |
| pick the right model for my hardware and task | [Choose a model](models.md) |
| run a server that existing TypeSafe Jev clients can call | [Serve it](guides/serve.md) |
| use LM Studio, Ollama or vLLM as the engine | [LM Studio, Ollama and vLLM](guides/model-servers.md) |
| automate the decisions the model is sure about and send the rest to a person | [Automate the confident decisions](guides/confident-automation.md) |
| check an AI agent's next step before it runs | [Agent guardrails](guides/agent-guardrails.md) |
| let Claude, Cursor or another AI assistant call it as a tool | [AI assistants (MCP)](guides/mcp.md) |
| copy a working script | [Examples and notebook](examples.md) |
| see how it compares with Jev, Laya and frontier LLMs | [Benchmarks](benchmarks.md) |
| know where it is weak | [Limitations](limitations.md) |
| see what is coming, or help build it | [Roadmap](roadmap.md) |
| look up a function, an endpoint or a flag | [Python API](reference/python-api.md) · [HTTP API](reference/http-api.md) · [Command line](reference/cli.md) |

## Why OpenDecider

- **Typed answers, not text.** Every answer is a full probability distribution you can threshold, route on or log.
- **Calibrated.** Distilled from two openly licensed teachers whose probabilities were temperature-scaled on held-out
  gold labels. Calibration error on 200 general decisions is 0.083–0.110, against 0.164 for TypeSafe Jev.
- **Runs where you are.** CPU, NVIDIA and Apple Silicon from Python; LM Studio, Ollama and vLLM as engines; a server
  that speaks Jev's `/v1/systemone` protocol, with Docker images.
- **Measured head to head.** Every model, including TypeSafe Jev through its own API, answered the same questions and
  was scored by the same code. [Every number](https://github.com/manjunathshiva/opendecider/blob/main/COMPARISON.md)
  can be rebuilt from the committed results.

| | typed-decisions (2,000 business decisions) | 200 general decisions | median latency |
|---|---|---|---|
| **opendecider-nano** (~400M) | **0.796** | 0.680 | **17 ms** |
| **opendecider-small** (4B) | 0.672 (zero-shot) | 0.735 | 40 ms |
| **opendecider-medium-td** (30B MoE) | 0.788 | **0.765** | 214 ms |
| **opendecider-large-td** (80B MoE) | **0.801** | 0.750 | 440 ms |
| TypeSafe Jev 1.13 (API) | 0.754 (zero-shot) | 0.730 | 404 ms |
| Laya's typed-decisions checkpoint | 0.766 | 0.570 | 21 ms |

nano, medium-td, large-td and Laya's checkpoint were fine-tuned on the typed-decisions train split (the test split was
never used); Jev and small are zero-shot there. See [Benchmarks](benchmarks.md) for the full picture, including where
Jev and Laya lead.

## Links

[GitHub](https://github.com/manjunathshiva/opendecider) ·
[PyPI](https://pypi.org/project/opendecider/) ·
[Models on Hugging Face](https://huggingface.co/collections/manjunathshiva/opendecider-6ab8c838909092518d50a9ea) ·
[Live demo](https://huggingface.co/spaces/manjunathshiva/opendecider-demo) ·
[Changelog](https://github.com/manjunathshiva/opendecider/blob/main/CHANGELOG.md)
