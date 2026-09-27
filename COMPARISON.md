# OpenDecider: full comparison

Every number below is read from the benchmark result files. All models answered the same questions and were scored by the same code. Jev was called through TypeSafe's own API; frontier LLMs through their APIs (OpenRouter) with option probabilities from token log-probabilities, as in our earlier [Jev vs frontier LLMs benchmark](https://github.com/manjunathshiva/jev-frontier-bench).

## 1. General decisions: 200 items none of the small models trained on

BANKING77 intent (78 options), BoolQ yes/no, Yelp 1–5 star rating, ChaosNLI (100 human labels per item). Accuracy is pooled over the 200 items. ECE: calibration error (lower is better). JSD: distance from the human label distribution on ChaosNLI (lower is better). Cost: USD per 1,000 decisions through the API (OpenDecider, Laya and CLM run on your own hardware).

| Model | Size / access | Accuracy | route | yes/no | rating | ambig | ECE | JSD vs humans | median latency | $ / 1k decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| **OpenDecider-medium-td** | 30B-A3B, open | **0.765** | 0.72 | 0.94 | 0.64 | 0.76 | 0.110 | 0.035 | 214 ms 4× L40S | local |
| **OpenDecider-small** | 4B, open | **0.735** | 0.70 | 0.90 | 0.60 | 0.74 | 0.087 | 0.040 | 40 ms L40S · 217 ms M4 Max | local |
| **OpenDecider-small-td** | 4B, open | **0.715** | 0.68 | 0.88 | 0.60 | 0.70 | 0.107 | 0.040 | 40 ms | local |
| **OpenDecider-nano** | ~400M, open | **0.680** | 0.68 | 0.74 | 0.64 | 0.66 | 0.092 | 0.045 | 17 ms L40S · 18 ms M4 Max | local |
| Claude Fable 5.1 | API | **0.840** | 0.88 | 0.94 | 0.74 | 0.80 | 0.064 | 0.043 | 4.27 s | $11.812 |
| GPT-6 Astra | API | **0.790** | 0.86 | 0.96 | 0.68 | 0.66 | 0.119 | 0.158 | 2.22 s | $6.963 |
| DeepSeek V4.1 Flash | API | **0.760** | 0.86 | 0.96 | 0.68 | 0.54 | 0.138 | 0.168 | 4.08 s | $0.158 |
| MiniMax M3 | API | **0.755** | 0.80 | 0.90 | 0.60 | 0.72 | 0.112 | 0.107 | 1.02 s | $0.149 |
| Kimi K3 | API | **0.745** | 0.80 | 0.90 | 0.68 | 0.60 | 0.119 | 0.127 | 6.28 s | $3.645 |
| Jev 1.13 (TypeSafe) | API | **0.730** | 0.76 | 0.94 | 0.62 | 0.60 | 0.164 | 0.148 | 404 ms | $0.025 |
| Qwen3-Next-80B-A3B, untrained | 80B, open | **0.750** | 0.68 | 0.92 | 0.64 | 0.76 | 0.230 | 0.225 | 310 ms | local |
| Qwen3-30B-A3B, untrained | 30B, open | **0.745** | 0.74 | 0.92 | 0.56 | 0.76 | 0.233 | 0.236 | 173 ms | local |
| Qwen3-4B, untrained (small's base) | 4B, open | **0.700** | 0.74 | 0.88 | 0.56 | 0.62 | 0.289 | 0.277 | 118 ms | local |
| Laya, typed-decisions checkpoint | ~400M, open | **0.570** | 0.40 | 0.84 | 0.38 | 0.66 | 0.162 | 0.111 | 21 ms | local |
| Laya | ~400M, open | **0.545** | 0.38 | 0.80 | 0.32 | 0.68 | 0.327 | 0.174 | 22 ms | local |
| CLM-8B (Contrastive-LM) | 8B, open | **0.400** | 0.02 | 0.60 | 0.38 | 0.60 | 0.106 | 0.122 | ~35 ms L40S | local |

The frontier LLMs are more accurate, at 25–370× the latency and a per-call bill. OpenDecider-small beats Jev and Laya, and is the best-calibrated model you can run yourself (ECE 0.087; only Claude Fable, 0.064, is lower). OpenDecider-medium-td is the most accurate model you can run yourself (0.765) and the closest of all systems to the human label spread (JSD 0.035). Distillation moved Qwen3-4B from 0.700 to 0.735 accuracy (calibration error 0.289 to 0.087) and Qwen3-30B-A3B from 0.745 to 0.765 (0.233 to 0.110).

Two extra tasks (100 items each), for the models run on them:

| Model | AG News (4 topics) | DAIR Emotion (6) |
|---|---|---|
| OpenDecider-medium-td | 0.85 | 0.65 |
| OpenDecider-small | 0.84 | 0.65 |
| OpenDecider-small-td | 0.85 | 0.62 |
| OpenDecider-nano | 0.73 | 0.59 |
| Jev 1.13 | 0.85 | 0.66 |
| Laya | 0.93 | 0.63 |
| Laya typed-decisions | 0.92 | 0.62 |
| CLM-8B | 0.42 | 0.26 |

## 2. typed-decisions (LocalLLaMA/typed-decisions test split, 2,000 decisions)

Scored with the [Antz AI Jev-vs-Laya harness](https://github.com/pavanjava/jev_and_laya_benchmarking) (its per-question CSV joined with ours; 0 gold-label mismatches). 95% CIs: paired bootstrap over cases.

| Model | Training on typed-decisions | Accuracy | choice | score | yes/no | vs Laya-td (95% CI) |
|---|---|---|---|---|---|---|
| **OpenDecider-nano** | train split (like Laya-td) | **0.796** | 0.762 | 0.769 | 0.867 | +0.030 [+0.014, +0.044] |
| **OpenDecider-small-td** | train split | **0.792** | 0.755 | 0.769 | 0.860 | +0.026 [+0.008, +0.043] |
| **OpenDecider-medium-td** | train split | **0.788** | 0.733 | 0.762 | 0.878 | +0.022 [+0.005, +0.040] |
| Laya, typed-decisions checkpoint | train split | **0.766** | 0.733 | 0.723 | 0.857 | – |
| Jev 1.13 (our run, 2026-09-26) | none | **0.754** | 0.737 | 0.701 | 0.843 | -0.012 [-0.034, +0.009] |
| Jev 1.13 (Antz AI's run) | none | **0.734** | 0.730 | 0.699 | 0.783 | -0.032 [-0.054, -0.012] |
| **OpenDecider-small** | none (zero-shot) | **0.671** | 0.648 | 0.631 | 0.748 | -0.095 [-0.120, -0.071] |
| CLM-8B | none | **0.358** | 0.253 | 0.330 | 0.500 | -0.408 [-0.435, -0.382] |
| Laya (English checkpoint), zero-shot | none | 0.362 | | | | |

## 3. Laya's own application battery (10 tasks × 400, Laya's script and cases)

Laya was trained on the first five datasets (its own `in_training` flags). OpenDecider trained on none of the ten; Jev's training data is not published.

| Model | all 10 | Laya-trained 5 | other 5 | AG News | support triage | Enron spam | phishing | RAG relevance | emotion | BANKING77 (77) | jailbreak | toxicity | model routing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OpenDecider-medium-td** | **0.725** | 0.682 | 0.768 | 0.882 | 0.378 | 0.950 | 0.652 | 0.550 | 0.555 | 0.785 | 0.762 | 0.802 | 0.935 |
| **OpenDecider-small** | **0.702** | 0.661 | 0.743 | 0.863 | 0.347 | 0.922 | 0.630 | 0.545 | 0.570 | 0.748 | 0.775 | 0.772 | 0.852 |
| **OpenDecider-small-td** | **0.703** | 0.662 | 0.743 | 0.860 | 0.347 | 0.920 | 0.642 | 0.542 | 0.570 | 0.693 | 0.772 | 0.812 | 0.870 |
| **OpenDecider-nano** | **0.656** | 0.655 | 0.656 | 0.810 | 0.435 | 0.865 | 0.630 | 0.535 | 0.570 | 0.645 | 0.905 | 0.650 | 0.511 |
| Jev 1.13 (TypeSafe) | **0.774** | 0.745 | 0.803 | 0.865 | 0.360 | 0.985 | 0.897 | 0.618 | 0.593 | 0.845 | 0.940 | 0.665 | 0.975 |
| Laya, typed-decisions ckpt | **0.702** | 0.796 | 0.609 | 0.953 | 0.505 | 0.958 | 0.940 | 0.625 | 0.600 | 0.492 | 0.762 | 0.530 | 0.659 |
| Laya (English) | **0.695** | 0.810 | 0.579 | 0.950 | 0.502 | 0.993 | 0.980 | 0.625 | 0.595 | 0.425 | 0.708 | 0.530 | 0.639 |
| Laya multilingual | **0.645** | 0.819 | 0.472 | 0.930 | 0.522 | 0.993 | 0.993 | 0.657 | 0.530 | 0.425 | 0.755 | 0.525 | 0.123 |
| CLM-8B | **0.431** | 0.438 | 0.425 | 0.405 | 0.133 | 0.627 | 0.420 | 0.603 | 0.292 | 0.000 | 0.713 | 0.505 | 0.614 |

## Notes

- OpenDecider-nano was fine-tuned on the typed-decisions **train** split, as Laya's typed-decisions checkpoint was; the test split was never used for training or model selection. OpenDecider-small-td and -medium-td had the same short fine-tune (100 train cases held out for model selection); OpenDecider-small never saw typed-decisions.
- No benchmark dataset above (or its family) is in OpenDecider's training data; every training pool was checked for text overlap with all test sets (0 overlaps).
- Frontier-LLM numbers come from the same 200 items, run 2026-09-19; Jev's typed-decisions score rose 2 points between Antz AI's run and ours (a newer Jev version).
- CLM-8B was run with its own engine through the official vLLM pooling server. Its model card's example (billing 0.93878) did not reproduce even that way (0.9889).
- Latencies: medians over the benchmark questions, one question per call. API latencies include the network.
