<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/manjunathshiva/opendecider/main/assets/logo-lockup-dark.png" />
    <img src="https://raw.githubusercontent.com/manjunathshiva/opendecider/main/assets/logo-lockup.png" alt="OpenDecider" width="400" />
  </picture>
</p>

**Open, calibrated System 1 decision models.** Ask typed questions (`choice`, `score`, `noul`) about any state (text, email, ticket or JSON) and get a calibrated probability for every option: 17 ms on an NVIDIA GPU, 18 ms on a Mac. Distilled from open teachers, and benchmarked head to head against TypeSafe Jev, Laya, CLM-8B and frontier LLMs on the same questions with the same scorer.

<div align="center">

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-opendecider--nano-blue)](https://huggingface.co/manjunathshiva/opendecider-nano)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-opendecider--small-blue)](https://huggingface.co/manjunathshiva/opendecider-small)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-OpenDecider-orange)](https://huggingface.co/collections/manjunathshiva/opendecider-6ab8c838909092518d50a9ea)
[![PyPI version](https://img.shields.io/pypi/v/opendecider.svg)](https://pypi.org/project/opendecider/)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Space-live%20demo-orange)](https://huggingface.co/spaces/manjunathshiva/opendecider-demo)
[![Full comparison](https://img.shields.io/badge/benchmarks-COMPARISON.md-2ea44f)](https://github.com/manjunathshiva/opendecider/blob/main/COMPARISON.md)
[![Platforms](https://img.shields.io/badge/runs%20on-CPU%20%C2%B7%20NVIDIA%20%C2%B7%20Apple%20Silicon-lightgrey)](https://github.com/manjunathshiva/opendecider#installation-details)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## Installation

```bash
pip install opendecider              # opendecider-nano
pip install "opendecider[small]"     # adds peft for opendecider-small
```

Python 3.10 or newer. Works on Linux, Windows and macOS, on CPU, NVIDIA (CUDA) and Apple Silicon (MPS), and picks the
device for you. Platform notes: [Installation details](https://github.com/manjunathshiva/opendecider#installation-details).
Try it without installing: [live demo](https://huggingface.co/spaces/manjunathshiva/opendecider-demo).

## Quickstart

```python
from opendecider import load

model = load("manjunathshiva/opendecider-nano")   # 0.8 GB, downloaded on first use; "…/opendecider-small" for the 4B

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

result = model.system_one(state, questions)
print(result["answers"]["department"]["choice"])   # billing        (probability 0.927)
print(result["answers"]["urgency"]["score"])       # 2 = blocking   (probability 0.604)
print(result["answers"]["churn_risk"]["noul"])     # 0.922 = probability the answer is yes
```

A state can be plain text or any JSON-serialisable object: a ticket with subject, body and customer fields, a log
record, an agent's tool-call trace. Questions can also be written with the helper classes `Choice`, `Score` and `Noul`.

## Beats Laya and Jev on typed-decisions

On the [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) benchmark (2,000 decisions across four
business workflows), scored with the [Antz AI Jev-vs-Laya harness](https://github.com/pavanjava/jev_and_laya_benchmarking),
**opendecider-nano scores 0.796**, against **0.766** for Laya's typed-decisions checkpoint (+0.030, 95% CI +0.014 to
+0.044) and **0.754** for TypeSafe Jev. Like Laya's checkpoint, it was fine-tuned on the dataset's train split; the test
split was never used for training or model selection. **opendecider-small, which never saw the dataset, scores 0.672**,
against 0.362 for Laya's base checkpoint.

## What's new in 0.1.0

* **First release:** `opendecider-nano` (~400M, Ettin encoder) and `opendecider-small` (4B, LoRA on Qwen3-4B-Instruct-2507).
* **Measured against TypeSafe Jev directly,** through TypeSafe's own API, on every benchmark, alongside Laya, CLM-8B and five frontier LLMs.
* **Same results on every platform:** both models give identical benchmark scores on Apple Silicon (MPS) and Linux + NVIDIA (CUDA).
* **Licence-clean data:** every training dataset is listed in [NOTICE](https://github.com/manjunathshiva/opendecider/blob/main/NOTICE). A non-commercial dataset found in our audit was removed before release, and no outputs of Claude or GPT models were used.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/manjunathshiva/opendecider/main/assets/opendecider_vs_jev_laya.png" alt="OpenDecider versus TypeSafe Jev, Laya, CLM-8B and frontier LLMs: typed-decisions accuracy, accuracy versus latency on 200 general decisions, Laya's own application battery, and calibration" width="100%" />
</p>

OpenDecider answers typed questions over any state in **a single forward pass** (nano) or a single next-token read
(small). There's no text generation, so nothing to parse and nothing to hallucinate. Every answer carries a full,
calibrated probability distribution you can threshold, route on or log.

Two checkpoints:

| | backbone | params | context | memory | use it for |
|---|---|---|---|---|---|
| [`opendecider-nano`](https://huggingface.co/manjunathshiva/opendecider-nano) | Ettin-encoder-400m | ~400M | 2,048 | 2.0 GiB | speed: 17–18 ms per question, ~9 ms batched; typed business decisions |
| [`opendecider-small`](https://huggingface.co/manjunathshiva/opendecider-small) | Qwen3-4B-Instruct-2507 + LoRA | 4B | 768 (training inputs) | 8.9 GiB, fits a 16 GB Mac | accuracy and calibration on decisions it has never seen |

### Coming next (in development)

* **opendecider-medium** (Qwen3-30B-A3B, for 36 GB+ machines) and **opendecider-large** (Qwen3-Next-80B-A3B, for 64 GB),
  aimed at closing the remaining gap to frontier LLMs.
* **MLX builds** for Apple Silicon, a **Colab notebook** for NVIDIA, and the **benchmark harness** as runnable scripts.

## Installation details

```bash
# 1. a virtual environment (macOS / Linux)
python3 -m venv .venv && source .venv/bin/activate
# Windows PowerShell:  py -m venv .venv ; .venv\Scripts\Activate.ps1

# 2. PyTorch for your hardware (skip if already installed)
pip install torch                                                        # macOS (Apple Silicon uses MPS) and CPU
pip install torch --index-url https://download.pytorch.org/whl/cu128      # Linux / Windows with an NVIDIA GPU

# 3. OpenDecider
pip install "opendecider[small]"
```

* **Device:** CUDA, then MPS, then CPU, chosen automatically. Override with `load(..., device="cpu")`.
* **Offline or air-gapped:** download a model folder once (`huggingface-cli download manjunathshiva/opendecider-nano --local-dir ./nano`), then `load("./nano")`.
* **CPU only:** nano runs fine on CPU for batch jobs. small needs ~17 GB of RAM in fp32 and is slow on CPU.
* **Memory:** nano 2.0 GiB, small 8.9 GiB of GPU or unified memory (measured on Apple Silicon).

## Decision primitives

```python
from opendecider import Choice, Score, Noul

Choice("Which team?", {"billing": "charges, refunds", "technical": "bugs"})   # pick one; descriptions optional
Choice("Which intent?", ["refund", "replacement", "information"])             # a plain list of labels
Score("How urgent?", ["not urgent", "soon", "blocking"])                      # ordered levels, lowest first
Noul("Is this spam?")                                                          # yes / no
Noul("Is this spam?", {"true": "unsolicited marketing", "false": "mail the user wants"})
```

Answers:

```python
{"type": "choice", "choice": "billing", "probabilities": {"billing": 0.927, ...}, "confidence": 0.927}
{"type": "score",  "score": 2, "expected": 1.51, "probabilities": {"0": 0.091, "1": 0.305, "2": 0.604}, "confidence": 0.604}
{"type": "noul",   "noul": 0.922, "probabilities": {"true": 0.922, "false": 0.078}, "confidence": 0.922}
```

`system_one(state, questions)` takes any number of questions about one state. opendecider-nano answers all of them in
one padded batch (9.3 ms per question at 50 questions on a Mac); opendecider-small answers them in sequence.

Measure latency on your own hardware: `python -m opendecider.bench_speed manjunathshiva/opendecider-nano`.

## Architecture

* **opendecider-nano:** Ettin-encoder-400m (bidirectional, fully fine-tuned) reads
  `question: …, [MASK] option 1, [MASK] option 2, …, input: <state>`. The hidden state at each `[MASK]` goes through a
  small MLP (Linear–GELU–LayerNorm–Linear) to one logit, then a softmax across that question's options. The answer space
  is defined at request time, so new schemas need no retraining. There's no per-option token budget, so a 78-option
  question costs one forward pass.
* **opendecider-small:** Qwen3-4B-Instruct-2507 with a LoRA adapter (r = 16, all linear projections). The options are
  lettered, and one forward pass gives the probability of each letter as the next token. Above 26 options it scores each
  option name's log-probability after the shared prompt.

## Training

**Distillation from calibrated teachers.** Two openly licensed teachers, Qwen3-235B-A22B-Instruct-2507 (Apache-2.0) and
DeepSeek V4.1 Flash (MIT), scored every training question through token log-probabilities. Each teacher was
temperature-scaled on held-out gold labels before the two were averaged, so the students learn calibrated distributions,
not hard labels. Datasets that come with gold labels only use label-smoothed gold.

**Data.** Public classification, intent, emotion, NLI, reading-comprehension, topic, toxicity, spam, relevance and
paraphrase datasets, plus synthetic business cases, emails and product reviews written for this project (full list and
licences in [NOTICE](https://github.com/manjunathshiva/opendecider/blob/main/NOTICE)). opendecider-nano then had a short fine-tune on the typed-decisions train split.
**No benchmark dataset below, or its family, is in the training data**, and every training pool was checked for text
overlap with all test sets (0 overlaps).

## Benchmarks

Every model answered the same questions and was scored by the same code. **TypeSafe Jev was measured directly through
TypeSafe's own API**, not taken from published figures. Full tables, per-task results and methodology: [COMPARISON.md](https://github.com/manjunathshiva/opendecider/blob/main/COMPARISON.md).

### Speed

| questions per call | nano, NVIDIA L40S | nano, Apple M4 Max | small, NVIDIA L40S | small, Apple M4 Max |
|---|---|---|---|---|
| 1 | 16.1 ms | 18.1 ms | 37.6 ms | 141 ms |
| 5 | 24.4 ms (4.9 ms/q) | 54.3 ms (10.9 ms/q) | 190.1 ms (38.0 ms/q) | 680 ms (136 ms/q) |
| 10 | 42.9 ms (4.3 ms/q) | 98.1 ms (9.8 ms/q) | 388.2 ms (38.8 ms/q) | 1.37 s (137 ms/q) |
| 50 | 189.5 ms (3.8 ms/q) | 467 ms (9.3 ms/q) | 1.94 s (38.7 ms/q) | 6.86 s (137 ms/q) |

For reference, TypeSafe Jev answered at a **404 ms** median per question through its API in our runs.

### OpenDecider vs TypeSafe Jev (measured through TypeSafe's API)

| Benchmark / metric | TypeSafe Jev 1.13 | opendecider-nano | opendecider-small |
|---|---|---|---|
| typed-decisions, 2,000 decisions | 0.754 | **0.796** | 0.672 (zero-shot) |
| 200 general decisions (BANKING77, BoolQ, Yelp, ChaosNLI) | 0.730 | 0.680 | **0.735** |
| Laya's application battery, 10 tasks | **0.774** | 0.656 | 0.702 |
| Calibration error (ECE), general decisions | 0.164 | 0.092 | **0.087** |
| Distance from the human label spread (ChaosNLI JSD) | 0.148 | 0.045 | **0.040** |
| Median latency, 1 question | 404 ms (API) | **17 ms** (L40S) | 40 ms (L40S) |
| Weights | closed API | **Apache-2.0** | **Apache-2.0** |
| Cost | $0.025 per 1,000 decisions | self-hosted | self-hosted |

#### Where Jev leads

* **Laya's application battery:** Jev 0.774 vs 0.702 (small) and 0.656 (nano); 0.803 on the five tasks Laya was not
  trained on. Jev is strongest on phishing (0.897, vs our 0.63), spam (0.985), model routing (0.975) and 77-label
  BANKING77 (0.845).
* **BoolQ-style yes/no reading questions** (0.94, vs 0.74 nano and 0.90 small) and **BANKING77 routing with 78 options**
  on our bench (0.76, vs 0.68 and 0.70).
* **typed-decisions without fine-tuning:** Jev 0.754 vs opendecider-small 0.672. The fine-tuned nano (0.796) passes it.

Where OpenDecider leads Jev: typed-decisions after fine-tuning (0.796 vs 0.754), general decisions (small 0.735 vs
0.730), calibration (ECE 0.087–0.092 vs 0.164), agreement with human label spread (JSD 0.040–0.045 vs 0.148), latency
(17–40 ms vs 404 ms), open weights and self-hosting.

### OpenDecider vs Laya

| Benchmark | Laya | Laya typed-decisions | opendecider-nano | opendecider-small |
|---|---|---|---|---|
| typed-decisions (Antz harness) | 0.362 | 0.766 | **0.796** | 0.672 |
| 200 general decisions | 0.545 | 0.570 | 0.680 | **0.735** |
| Laya's battery, all 10 tasks | 0.695 | **0.702** | 0.656 | **0.702** |
| Laya's battery, the 5 tasks Laya was not trained on | 0.579 | 0.609 | 0.656 | **0.743** |
| BANKING77, 77 labels (Laya's battery) | 0.425 | 0.492 | 0.645 | **0.748** |
| Calibration error (ECE), general decisions | 0.327 | 0.162 | 0.092 | **0.087** |

#### Where Laya leads

* **The five datasets Laya was trained on:** AG News 0.95, Enron spam 0.99, phishing 0.98, MS MARCO relevance 0.63,
  support triage 0.50. OpenDecider did not train on any of them.
* **Multilingual:** Laya has a 100+ language checkpoint and a router. OpenDecider is evaluated in English only.
* **Single-question speed on short inputs:** Laya is in the same range as nano (21–33 ms).

### Frontier LLMs, CLM-8B and untrained baselines (same 200 general decisions)

| Model | accuracy | ECE | median latency | $ / 1,000 decisions |
|---|---|---|---|---|
| Claude Fable 5.1 | **0.840** | **0.064** | 4.27 s | $11.81 |
| GPT-6 Astra | 0.790 | 0.119 | 2.22 s | $6.96 |
| DeepSeek V4.1 Flash | 0.760 | 0.138 | 4.08 s | $0.158 |
| MiniMax M3 | 0.755 | 0.112 | 1.02 s | $0.149 |
| Qwen3-Next-80B-A3B, untrained | 0.750 | 0.230 | local | – |
| Kimi K3 | 0.745 | 0.119 | 6.28 s | $3.64 |
| **opendecider-small** | **0.735** | **0.087** | **40 ms** | self-hosted |
| TypeSafe Jev 1.13 | 0.730 | 0.164 | 404 ms | $0.025 |
| Qwen3-4B-Instruct-2507, untrained (small's base) | 0.700 | 0.289 | local | – |
| **opendecider-nano** | **0.680** | **0.092** | **17 ms** | self-hosted |
| CLM-8B (Contrastive-LM) | 0.400 | 0.106 | ~35 ms | self-hosted |

Frontier LLMs are more accurate, at 25–370× the latency and with a per-call bill. Distillation moved Qwen3-4B from 0.700
to 0.735 and cut its calibration error from 0.289 to 0.087. CLM-8B, a contrastive reranker, is near chance on
classification-style decisions (0.000 on label-only BANKING77); its strongest task is passage relevance (0.603 on MS
MARCO, near Laya's 0.625, which trained on it).

## Honest limits

* **Phishing detection is the weakest task:** 0.63 on Laya's battery for both models, against Jev's 0.90 and Laya's 0.98 (Laya trained on that dataset).
* **TypeSafe Jev leads Laya's application battery** (0.774 vs 0.702 small, 0.656 nano).
* **opendecider-nano trails Laya on Laya's battery overall** (0.656 vs 0.695), because half its tasks are Laya's training data.
* **opendecider-small is zero-shot on typed-decisions** and trails Jev there (0.672 vs 0.754).
* **English only so far.** The training data includes some Spanish, German, French, Portuguese, Italian and Dutch, but no multilingual evaluation has been run.
* **opendecider-small answers questions one at a time** (~137 ms per question on a Mac, ~40 ms on an L40S). Use nano when you need many decisions per second.
* **Descriptions help.** Very terse or cryptic option labels are harder for every model, so give options a short description when you can.

## Links

* **Live demo:** https://huggingface.co/spaces/manjunathshiva/opendecider-demo
* **PyPI:** https://pypi.org/project/opendecider/
* **Models:** [opendecider-nano](https://huggingface.co/manjunathshiva/opendecider-nano) · [opendecider-small](https://huggingface.co/manjunathshiva/opendecider-small) · [collection](https://huggingface.co/collections/manjunathshiva/opendecider-6ab8c838909092518d50a9ea)
* **Full benchmark tables:** [COMPARISON.md](https://github.com/manjunathshiva/opendecider/blob/main/COMPARISON.md)
* **Related:** [Jev vs frontier LLMs benchmark](https://github.com/manjunathshiva/jev-frontier-bench)

## License

Code and weights: Apache-2.0. Base models: Ettin-encoder-400m (MIT), Qwen3-4B-Instruct-2507 (Apache-2.0).
Training-data attributions: [NOTICE](https://github.com/manjunathshiva/opendecider/blob/main/NOTICE).

```bibtex
@software{janardhan2026opendecider,
  title  = {OpenDecider: open, calibrated System 1 decision models},
  author = {Manjunath Janardhan},
  year   = {2026},
  url    = {https://github.com/manjunathshiva/opendecider}
}
```
