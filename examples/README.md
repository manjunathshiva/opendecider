# Examples

Short scripts, one per use case. Each runs on a laptop CPU with
[opendecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano) (a 0.8 GB download on first use) unless
noted. CI runs each of them against the released model whenever the package or the examples change, except
`remote_backends.py`, which needs LM Studio, Ollama or vLLM running and is only compile-checked there.

| script | what it shows | run it |
|---|---|---|
| [`quickstart.py`](quickstart.py) | three typed questions (`choice`, `score`, `noul`) about one ticket | `python examples/quickstart.py` |
| [`support_triage.py`](support_triage.py) | a queue of tickets in one call: route by team, flag churn risk, send unsure answers to a person | `python examples/support_triage.py` |
| [`agent_guardrail.py`](agent_guardrail.py) | check an AI agent's JSON trace before its next step: continue, retry, ask the user or stop | `python examples/agent_guardrail.py` |
| [`confident_automation.py`](confident_automation.py) | on 2,000 labelled business decisions: how many you can automate at a given accuracy | `python examples/confident_automation.py` |
| [`serve_client.py`](serve_client.py) | call `opendecider serve` over HTTP (Jev's `/v1/systemone` protocol), with retries | start the server, then `python examples/serve_client.py` |
| [`remote_backends.py`](remote_backends.py) | the same decision through LM Studio, Ollama or vLLM | `python examples/remote_backends.py lmstudio:opendecider-small` |

```bash
pip install opendecider                    # nano
pip install "opendecider[serve]"           # serve_client.py: the server
pip install pandas pyarrow                 # confident_automation.py: reads the typed-decisions test split
pip install "opendecider[small]"           # the 4B models: pass e.g. manjunathshiva/opendecider-small-td as the model
```

The scripts that take a model name accept any OpenDecider model (`quickstart.py`, `support_triage.py` and
`agent_guardrail.py` as the first argument, `confident_automation.py` as `--model`).

## What `confident_automation.py` prints for nano

The full test split (400 cases, 2,000 decisions; no OpenDecider model was trained on it), on a laptop CPU in about a
minute:

```
manjunathshiva/opendecider-nano: 400 cases, 2000 decisions, accuracy 0.796

automate when p >= | automated | accuracy of automated | to a person
              0.0 |     100% |                 0.796 |         0%
              0.5 |      76% |                 0.873 |        24%
              0.6 |      51% |                 0.941 |        49%
              0.7 |      32% |                 0.972 |        68%
              0.8 |      21% |                 0.998 |        79%
              0.9 |      13% |                 0.996 |        87%

most confident | accuracy
          100% | 0.796
           70% | 0.894
           50% | 0.943
```

Automating the answers with p ≥ 0.6 handles half the decisions at 94% accuracy; the rest go to a person. The gold
labels come from a teacher model and are themselves imperfect (the dataset card has the details), so read the top
rows as "agrees with the labels", not as a ceiling.

## Notebook

[`notebooks/opendecider_colab.ipynb`](../notebooks/opendecider_colab.ipynb) walks through the same ground on a free
Colab GPU, with the outputs from a real run:
[open it in Colab](https://colab.research.google.com/github/manjunathshiva/opendecider/blob/main/notebooks/opendecider_colab.ipynb).
