# OpenDecider

**Small, open decision models ("System One") that answer typed questions about your data with calibrated probabilities.**
Ask a *choice*, a *score* or a *yes/no* question about a ticket, email, log line or JSON record and get back
a probability for every option, fast enough to sit inside any pipeline or agent loop.
Apache-2.0, runs on CPU, NVIDIA GPUs and Apple Silicon.

Latencies are medians over our benchmark questions. Both models give identical benchmark results on Apple Silicon (MPS) and Linux + NVIDIA (CUDA).

| Model | Size | Built on | Speed (per question) | Best at |
|---|---|---|---|---|
| [**opendecider-nano**](https://huggingface.co/manjunathshiva/opendecider-nano) | ~400M, 0.8 GB | Ettin-encoder-400m | **17 ms** NVIDIA L40S · **18 ms** Apple M4 Max | speed; typed business decisions |
| [**opendecider-small**](https://huggingface.co/manjunathshiva/opendecider-small) | 4B (LoRA on Qwen3-4B), fits a 16 GB Mac | Qwen3-4B-Instruct-2507 | 40 ms NVIDIA L40S · ~200 ms Apple M4 Max | accuracy + calibration on unseen tasks |

```bash
pip install opendecider              # nano
pip install "opendecider[small]"     # small (adds peft)
```

```python
from opendecider import load, Choice, Score, Noul

model = load("manjunathshiva/opendecider-nano")
r = model.system_one(
    {"subject": "Charged twice!!", "body": "Card charged twice for order #1182. Fix this today or I cancel."},
    {
        "team":   Choice("Which team should handle this ticket?",
                         {"billing": "charges, invoices, refunds", "technical": "bugs, outages", "account": "login, cancellation"}),
        "urgent": Noul("Does this need a reply today?"),
        "churn":  Score("How likely is the customer to leave?", ["unlikely", "possible", "likely", "very likely"]),
    })
r["answers"]["team"]      # {'type': 'choice', 'choice': 'billing', 'probabilities': {'billing': 0.90, ...}, 'confidence': 0.90}
r["answers"]["urgent"]    # {'type': 'noul', 'noul': 0.89, 'probabilities': {'true': 0.89, 'false': 0.11}, ...}
r["answers"]["churn"]     # {'type': 'score', 'score': 3, 'expected': 2.25, 'probabilities': {'0': 0.07, ...}, ...}
```

The question format follows TypeSafe's Jev and Laya: `choice` (pick one option; options may carry descriptions),
`score` (an ordered rubric, lowest first) and `noul` (yes/no, optionally with descriptions of what true and false mean).

## How it compares

All models answered the same questions and were scored by the same code. Jev was called through TypeSafe's own API.

**typed-decisions** ([LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test split,
2,000 decisions), scored with the [Antz AI Jev-vs-Laya harness](https://github.com/pavanjava/jev_and_laya_benchmarking):

| Model | Accuracy | vs Laya typed-decisions (95% CI, paired bootstrap) |
|---|---|---|
| **opendecider-nano** (~400M) | **0.796** | **+0.030 [+0.014, +0.044]** |
| Laya, typed-decisions checkpoint (~400M) | 0.766 | – |
| Jev (TypeSafe API, 2026-09-26) | 0.755 | |
| opendecider-small (4B, zero-shot) | 0.672 | |
| CLM-8B (Contrastive-LM, zero-shot) | 0.358 | |

Like Laya's typed-decisions checkpoint, opendecider-nano was fine-tuned on this dataset's **train** split; the 400 test
cases were never used for training or model selection. opendecider-small never saw the dataset.

**General decisions on datasets none of these models trained on** (200 items: BANKING77 intent with 78 options, BoolQ yes/no,
Yelp star ratings, ChaosNLI; plus AG News and DAIR Emotion):

| Model | Accuracy | Calibration error (ECE, lower is better) |
|---|---|---|
| **opendecider-small** | **0.735** | 0.088 |
| Jev | 0.730 | 0.164 |
| **opendecider-nano** | 0.680 | 0.092 |
| Laya | 0.545 | 0.327 |
| CLM-8B | 0.400 | 0.106 |

**Laya's own application battery** (its `research/scripts/bench_apps.py`: 10 tasks × 400, same cases). Laya was trained
on 5 of these 10 datasets, so we also split the score:

| Model | All 10 tasks | 5 tasks Laya trained on | 5 tasks Laya did not train on |
|---|---|---|---|
| **opendecider-small** | **0.703** | 0.661 | **0.743** |
| Laya, typed-decisions checkpoint | 0.702 | 0.796 | 0.609 |
| Laya | 0.695 | 0.810 | 0.579 |
| opendecider-nano | 0.656 | 0.655 | 0.656 |
| CLM-8B | 0.431 | 0.438 | 0.425 |

### Where OpenDecider is weaker

- On Laya's battery, **Laya wins on the datasets it was trained on** (spam 0.99, phishing 0.98, AG News 0.95).
  Phishing (~0.63) is our weakest task.
- opendecider-nano trails Laya's battery overall (0.656 vs 0.695).
- Jev is still ahead of opendecider-small on typed-decisions without fine-tuning (0.755 vs 0.672).
- Evaluated in English only; the training data includes some Spanish, German, French, Portuguese, Italian and Dutch.
- opendecider-small is ~10x slower than nano; use it where accuracy on unfamiliar decisions matters more than latency.

## How it works

- **nano:** one encoder pass over `question, [MASK] option 1, [MASK] option 2, ..., state`. The hidden state at each
  marker goes through a small MLP, then a softmax across options. A 78-option question costs one forward pass.
- **small:** the options are lettered and the probability of each letter as the next token is read in one forward pass
  (label log-probabilities for more than 26 options).
- **Training:** distillation from calibrated soft labels. Two open teachers (Qwen3-235B-A22B-Instruct-2507, DeepSeek
  V4.1 Flash) scored the training questions through log-probabilities, and each teacher was temperature-scaled on
  held-out gold labels before averaging; datasets with gold labels only use label-smoothed gold. Data: public classification, NLI, QA, relevance, toxicity and paraphrase datasets plus
  synthetic business cases, emails and reviews (see [NOTICE](NOTICE)). Only openly licensed teachers were used; no outputs of proprietary models.
- **No test data in training:** every benchmark dataset above (and its family) is excluded, and each training pool is
  checked for text overlap against all test sets (0 overlaps).

## Licence

Code and weights: Apache-2.0. Base models: Ettin-encoder-400m (MIT), Qwen3-4B-Instruct-2507 (Apache-2.0).
Training-data attributions: [NOTICE](NOTICE).
