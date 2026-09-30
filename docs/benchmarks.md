# Benchmarks

Every model answered the same questions and was scored by the same code. **TypeSafe Jev was measured directly through
TypeSafe's own API**, not taken from published figures. Full tables, per-task results and methodology:
[COMPARISON.md](https://github.com/manjunathshiva/opendecider/blob/main/COMPARISON.md).

**Reproduce every number:** `python benchmarks/report.py` rebuilds all the tables from the committed results, and
`python benchmarks/run.py --model <name>` re-scores any model
([benchmarks/](https://github.com/manjunathshiva/opendecider/tree/main/benchmarks)).

![OpenDecider versus TypeSafe Jev, Laya, CLM-8B and frontier LLMs](https://raw.githubusercontent.com/manjunathshiva/opendecider/main/assets/opendecider_vs_jev_laya.png)

## The test sets

- **typed-decisions** ([dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)): 2,000 decisions across
  four business workflows, scored with the [Jev-vs-Laya harness](https://github.com/pavanjava/jev_and_laya_benchmarking)
  published by Kameshwara Pavan kumar Mantha and the Antz AI team. nano, small-td, medium-td, large-td and Laya's
  typed-decisions checkpoint were fine-tuned on its train split; the test split was never used for training or model
  selection. Zero-shot models are a reference there, not a head-to-head.
- **200 general decisions** none of the models trained on: BANKING77 intent (78 options), BoolQ yes/no, Yelp 1–5 star
  rating and ChaosNLI (100 human labels per item).
- **Laya's application battery:** Laya's own 10-task benchmark, at a pinned commit.

## OpenDecider vs TypeSafe Jev

| benchmark / metric | TypeSafe Jev 1.13 | nano | small | medium-td | large-td |
|---|---|---|---|---|---|
| typed-decisions, 2,000 decisions | 0.754 (zero-shot) | 0.796 | 0.672 (zero-shot) | 0.788 | **0.801** |
| 200 general decisions | 0.730 | 0.680 | 0.735 | **0.765** | 0.750 |
| Laya's application battery, 10 tasks | **0.774** | 0.656 | 0.702 | 0.725 | 0.718 |
| calibration error (ECE), general decisions ↓ | 0.164 | 0.092 | 0.087 | 0.110 | **0.083** |
| distance from the human label spread (ChaosNLI JSD) ↓ | 0.148 | 0.045 | 0.040 | 0.035 | **0.030** |
| median latency, 1 question | 404 ms (API) | **17 ms** (L40S) | 40 ms (L40S) | 214 ms (4× L40S) | 440 ms (4× L40S) |
| weights | closed API | **Apache-2.0** | **Apache-2.0** | **Apache-2.0** | **Apache-2.0** |

**Where Jev leads:** Laya's application battery (0.774, and 0.803 on the five tasks Laya was not trained on), phishing
(0.897 vs our 0.63–0.70), jailbreak detection (0.940), spam (0.985), model routing (0.975), BoolQ-style yes/no reading
(0.94) and typed-decisions without fine-tuning (0.754 vs small's 0.672).

## OpenDecider vs Laya

| benchmark | Laya | Laya typed-decisions | nano | small | medium-td | large-td |
|---|---|---|---|---|---|---|
| typed-decisions | 0.362 | 0.766 | 0.796 | 0.672 | 0.788 | **0.801** |
| 200 general decisions | 0.545 | 0.570 | 0.680 | 0.735 | **0.765** | 0.750 |
| Laya's battery, all 10 tasks | 0.695 | 0.702 | 0.656 | 0.702 | **0.725** | 0.718 |
| Laya's battery, the 5 tasks Laya was not trained on | 0.579 | 0.609 | 0.656 | 0.743 | **0.768** | 0.757 |
| calibration error (ECE), general decisions ↓ | 0.327 | 0.162 | 0.092 | 0.087 | 0.110 | **0.083** |

Like for like on typed-decisions (both fine-tuned on the train split), nano leads Laya's typed-decisions checkpoint by
+0.030 (95% CI +0.014 to +0.044). **Where Laya leads:** the five datasets it was trained on (AG News 0.95, Enron spam
0.99, phishing 0.98), multilingual use (a 100+ language checkpoint; OpenDecider is evaluated in English only), and
single-question speed on short inputs, where it is in the same range as nano.

## Frontier LLMs (same 200 general decisions)

| model | accuracy | ECE ↓ | median latency | $ / 1,000 decisions |
|---|---|---|---|---|
| Claude Fable 5.1 | **0.840** | **0.064** | 4.27 s | $11.81 |
| GPT-6 Astra | 0.790 | 0.119 | 2.22 s | $6.96 |
| **opendecider-medium-td** | 0.765 | 0.110 | 214 ms | self-hosted |
| DeepSeek V4.1 Flash | 0.760 | 0.138 | 4.08 s | $0.158 |
| MiniMax M3 | 0.755 | 0.112 | 1.02 s | $0.149 |
| **opendecider-large-td** | 0.750 | 0.083 | 440 ms | self-hosted |
| Kimi K3 | 0.745 | 0.119 | 6.28 s | $3.64 |
| **opendecider-small** | 0.735 | 0.087 | 40 ms | self-hosted |
| TypeSafe Jev 1.13 | 0.730 | 0.164 | 404 ms | $0.025 |
| **opendecider-nano** | 0.680 | 0.092 | 17 ms | self-hosted |
| CLM-8B (Contrastive-LM) | 0.400 | 0.106 | ~35 ms | self-hosted |

Only Claude Fable 5.1 and GPT-6 Astra beat opendecider-medium-td here, at 10–20× its latency and with a per-call bill.
The 200-item set is about ±3 points, so medium-td, DeepSeek V4.1 Flash and MiniMax M3 are close.

## Speed

| questions per call | nano, NVIDIA L40S | nano, Apple M4 Max | small, NVIDIA L40S | small, Apple M4 Max |
|---|---|---|---|---|
| 1 | 16.1 ms | 18.1 ms | 37.6 ms | 141 ms |
| 5 | 24.4 ms (4.9 ms/q) | 54.3 ms (10.9 ms/q) | 190.1 ms (38.0 ms/q) | 680 ms (136 ms/q) |
| 10 | 42.9 ms (4.3 ms/q) | 98.1 ms (9.8 ms/q) | 388.2 ms (38.8 ms/q) | 1.37 s (137 ms/q) |
| 50 | 189.5 ms (3.8 ms/q) | 467 ms (9.3 ms/q) | 1.94 s (38.7 ms/q) | 6.86 s (137 ms/q) |

Throughput under concurrent load is in [Serve it](guides/serve.md#performance-under-load).
