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

## Prompt guard

`opendecider.guard` against Laya's guard on the test splits of three public datasets (2,438 prompts):
[deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) (116, English and German),
[jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) (262) and
[xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) (2,060). Both
ask the same two questions, the attack questions of Laya's guard preset. Each model's threshold was set on the train
splits by a rule fixed before any test result was seen (the threshold that maximises balanced accuracy, averaged over
the three datasets) and applied once to test. No test prompt is in OpenDecider's training data (0 text overlaps).

| model | threshold | all (95% CI) | attacks caught | benign flagged | deepset | jailbreak | safeguard |
|---|---|---|---|---|---|---|---|
| **opendecider-small-td** | 0.484 | **0.936** (0.927–0.946) | 0.928 | **0.060** | **0.828** | 0.897 | **0.947** |
| opendecider-small | 0.500 | 0.900 (0.888–0.912) | 0.914 | 0.108 | 0.802 | 0.863 | 0.910 |
| Laya (English checkpoint) | 0.271 | 0.898 (0.886–0.910) | 0.934 | 0.121 | 0.767 | 0.969 | 0.897 |
| Laya (default router) | 0.271 | 0.897 (0.884–0.908) | 0.929 | 0.121 | 0.724 | **0.973** | 0.897 |
| opendecider-nano | 0.387 | 0.831 (0.816–0.845) | 0.914 | 0.214 | 0.776 | 0.901 | 0.825 |

Accuracy per dataset. Paired with Laya's English checkpoint, opendecider-small-td is ahead by 0.038 overall (95% CI
+0.023 to +0.052) and by 0.050 on safeguard, within noise on deepset (+0.060, CI −0.017 to +0.138), and **behind on
jailbreak-classification** (−0.073, CI −0.111 to −0.034), where it flags 21% of the benign prompts against Laya's 6.5%.
safeguard is 85% of the prompts, so it weighs most in the totals; averaged per dataset the two are close (balanced
accuracy 0.889 vs 0.884). Tuning the threshold on the train splits did not help on test: at 0.5, small-td scores 0.945
and Laya 0.909, slightly above their tuned results. The guard keeps the train-split thresholds, as the rule set out.

### Quantised builds

The published GGUF builds (run through Ollama) and MLX builds, on the same test splits. Each build's threshold was set
on the train splits by the same rule as its model's. A 4-bit build scores higher than its model, so at 0.5 it flags
far more benign prompts; at its own threshold it screens as well as the full model.

| build | threshold | all (95% CI) | attacks caught | benign flagged | at 0.5: all | at 0.5: benign flagged |
|---|---|---|---|---|---|---|
| opendecider-small-td (PyTorch, above) | 0.484 | 0.936 (0.927–0.946) | 0.928 | 0.060 | 0.945 | 0.040 |
| opendecider-small-td GGUF Q8_0 | 0.506 | 0.940 (0.931–0.950) | 0.925 | 0.052 | 0.938 | 0.057 |
| **opendecider-small-td GGUF Q4_K_M** | 0.557 | **0.957** (0.949–0.966) | 0.936 | **0.031** | 0.922 | 0.106 |
| opendecider-small (PyTorch, above) | 0.500 | 0.900 (0.888–0.912) | 0.914 | 0.108 | 0.900 | 0.108 |
| opendecider-small GGUF Q8_0 | 0.512 | 0.920 (0.910–0.932) | 0.903 | 0.070 | 0.911 | 0.091 |
| opendecider-small GGUF Q4_K_M | 0.558 | 0.934 (0.923–0.944) | 0.915 | 0.057 | 0.842 | 0.225 |
| opendecider-small MLX 8-bit | 0.516 | 0.913 (0.902–0.925) | 0.898 | 0.079 | 0.907 | 0.094 |
| opendecider-small MLX 4-bit | 0.547 | 0.906 (0.894–0.917) | 0.926 | 0.105 | 0.871 | 0.178 |

The guard uses each build's threshold when the model name says which build it is: an Ollama name with its tag
(`hf.co/manjunathshiva/opendecider-small-td-GGUF:Q4_K_M`), an LM Studio name with its variant
(`opendecider-small-td@q4_k_m`), or the MLX build's name. A name that does not say (an Ollama name without its tag,
an LM Studio model loaded without its variant in the name, or a GGUF given a plain name) gets its model's threshold
or 0.5, both at or below every build's, so it flags more rather than less. Averaged per dataset (balanced accuracy), the
builds' thresholds and 0.5 are within 0.005 of each other except small's Q4_K_M (0.885 vs 0.853); the totals move
more because two thirds of the test prompts are benign (1,589 of 2,438, 1,410 of them in safeguard), and a 4-bit
build's extra flags at 0.5 fall on them.

`python benchmarks/guard.py report` prints these tables from the committed results, and two more setups: one yes/no
question (Laya's published deepset question) and Laya's whole guard preset scored with LayaGuardrail's own rule. See
[Agent guardrails](guides/agent-guardrails.md#how-accurate-is-it) for what the numbers mean in practice.

## In the browser

`@opendecider/web` runs opendecider-nano as ONNX: q8 (8-bit weights, float32 elsewhere; WebAssembly's default) and q8f16
(8-bit weights, float16 elsewhere; WebGPU's default). Both were scored with native ONNX Runtime on the CPU, which gives
the same logits as the browser's WebAssembly to 1e-6 (WebGPU: within 0.02), against the PyTorch model's committed
answers. A build ships only if it gives PyTorch's answer on at least 99% of the questions and loses at most 0.5 points
on each benchmark.

| build | download | typed-decisions (2,000) | general (200) | Laya's battery (10 × 400) | answers as PyTorch |
|---|---|---|---|---|---|
| opendecider-nano (PyTorch, above) | 790 MB (bf16) | 0.796 | 0.680 | 0.656 | – |
| **q8** (WebAssembly) | 569 MiB | 0.795 | 0.680 | 0.656 | 99.6–99.8% |
| **q8f16** (WebGPU) | 450 MiB | 0.798 | 0.680 | 0.656 | 99.5–99.8% |

Not shipped: 4-bit builds (286–410 MiB) changed 2.5–4.3% of the answers (typed-decisions 0.790), whatever the
quantisation method (symmetric, asymmetric, HQQ), and dynamic int8 gave different logits in the browser's WebAssembly
than in native ONNX Runtime, so its benchmark would not describe what runs in the browser.

As a guard (the attack questions, each build at its own train-split threshold):

| build | threshold | all (95% CI) | attacks caught | benign flagged |
|---|---|---|---|---|
| opendecider-nano (PyTorch, above) | 0.387 | 0.831 (0.816–0.845) | 0.914 | 0.214 |
| q8 | 0.391 | 0.828 (0.813–0.843) | 0.914 | 0.218 |
| q8f16 | 0.390 | 0.828 (0.813–0.842) | 0.914 | 0.218 |

## Speed

| questions per call | nano, NVIDIA L40S | nano, Apple M4 Max | small, NVIDIA L40S | small, Apple M4 Max |
|---|---|---|---|---|
| 1 | 16.1 ms | 18.1 ms | 37.6 ms | 141 ms |
| 5 | 24.4 ms (4.9 ms/q) | 54.3 ms (10.9 ms/q) | 190.1 ms (38.0 ms/q) | 680 ms (136 ms/q) |
| 10 | 42.9 ms (4.3 ms/q) | 98.1 ms (9.8 ms/q) | 388.2 ms (38.8 ms/q) | 1.37 s (137 ms/q) |
| 50 | 189.5 ms (3.8 ms/q) | 467 ms (9.3 ms/q) | 1.94 s (38.7 ms/q) | 6.86 s (137 ms/q) |

Throughput under concurrent load is in [Serve it](guides/serve.md#performance-under-load).
