# OpenDecider benchmark harness

Reproduces every number in the main README and [COMPARISON.md](../COMPARISON.md).

```bash
pip install -r benchmarks/requirements.txt

# 1. Rebuild every table from the committed results (no model runs, a few minutes of downloads)
python benchmarks/report.py

# 2. Re-score a model yourself (resumable; writes benchmarks/results/<suite>/<name>.jsonl)
python benchmarks/run.py --model opendecider-nano
python benchmarks/run.py --model opendecider-small --suites general,typed
python benchmarks/run.py --model laya-td --suites general            # pip install laya
TYPESAFE_API_KEY=... python benchmarks/run.py --model jev             # TypeSafe's own API
python benchmarks/run.py --model ./my-model --name mine               # any OpenDecider folder
```

## The three benchmarks

| suite | what | source |
|---|---|---|
| `general` | 200 decisions: BANKING77 intent (78 options), BoolQ yes/no, Yelp 1–5 stars, ChaosNLI with 100 human labels per item (50 each); plus AG News and DAIR Emotion (100 each) | rebuilt from the public datasets with fixed seeds and checked against `manifests/` (SHA-256 of every state and question) |
| `typed` | typed-decisions test split: 400 cases, 2,000 decisions | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) `all/test`; compared with [Antz AI's per-question results](https://github.com/pavanjava/jev_and_laya_benchmarking) (pinned commit), paired bootstrap 95% CIs |
| `laya` | Laya's own application battery: 10 tasks × 400 | Laya's `research/scripts/bench_apps.py`, cloned at the pinned commit and run unchanged; Laya's own committed results are read from the same commit |

No dataset text is committed here: `items.py` rebuilds it, and `results/` holds only item ids and
probabilities. Metrics are the same code that produced the published numbers: accuracy, expected
calibration error (10 bins, top-option confidence) and Jensen-Shannon distance from ChaosNLI's human label
distribution; Laya's battery uses Laya's own `metrics()`.

## Committed results

| model | general | typed | laya | how it was run |
|---|---|---|---|---|
| opendecider-nano | ✅ | ✅ | ✅ | the released model through this package (identical on Apple Silicon and NVIDIA) |
| opendecider-small | ✅ | ✅ | ✅ | the released model through this package, NVIDIA L40S |
| jev | ✅ | ✅ | ✅ | TypeSafe's own API, Jev 1.13, 2026-09-26/27 |
| laya, laya-td | ✅ | Antz AI's run | Laya's committed run | `pip install laya` (0.3.x) for `general` |
| clm-8b | ✅ | ✅ | ✅ | Contrastive-LM/CLM-v0.1-8B with its own engine through the official vLLM pooling server |

Frontier LLM numbers (Claude Fable 5.1, GPT-6 Astra, DeepSeek V4.1 Flash, MiniMax M3, Kimi K3) come from the
same 200 general decisions in [jev-frontier-bench](https://github.com/manjunathshiva/jev-frontier-bench).

## Rules we held to

* No benchmark dataset above, or its family, is in OpenDecider's training data; every training pool was checked
  for text overlap with all test sets (0 overlaps).
* opendecider-nano was fine-tuned on the typed-decisions **train** split, like Laya's typed-decisions checkpoint;
  the test split was never used for training or model selection.
