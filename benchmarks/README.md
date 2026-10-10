# OpenDecider benchmark harness

Reproduces every number in the main README and [COMPARISON.md](../COMPARISON.md), and the prompt guard's
[benchmark against Laya](#the-prompt-guard-benchmark).

```bash
pip install -r benchmarks/requirements.txt

# 1. Rebuild every table from the committed results (no model runs, a few minutes of downloads)
python benchmarks/report.py

# 2. Re-score a model yourself (resumable; writes benchmarks/results/<suite>/<name>.jsonl)
python benchmarks/run.py --model opendecider-nano
python benchmarks/run.py --model opendecider-small --suites general,typed
python benchmarks/run.py --model laya-td --suites general            # pip install laya
TYPESAFE_API_KEY=... python benchmarks/run.py --model jev             # TypeSafe's own API
# Microsoft-Decision-1 through your own Microsoft Foundry deployment ("model" in each request is the deployment's name)
FOUNDRY_END_POINT=https://<resource>.services.ai.azure.com/providers/microsoft/v1/systemone \
FOUNDRY_API_KEY=... FOUNDRY_DEPLOYMENT=<deployment name> \
    python benchmarks/run.py --model foundry --name microsoft-decision-1
python benchmarks/run.py --model ./my-model --name mine               # any OpenDecider folder
# @opendecider/web's ONNX builds in native ONNX Runtime (pip install onnxruntime; the browser's WebAssembly gives the
# same logits): hf download manjunathshiva/opendecider-nano-ONNX --local-dir nano-onnx
python benchmarks/run.py --model onnx:nano-onnx/onnx/model_q8.onnx --name opendecider-nano-onnx-q8
```

## The benchmarks

| suite | what | source |
|---|---|---|
| `general` | 200 decisions: BANKING77 intent (78 options), BoolQ yes/no, Yelp 1–5 stars, ChaosNLI with 100 human labels per item (50 each); plus AG News and DAIR Emotion (100 each) | rebuilt from the public datasets with fixed seeds and checked against `manifests/` (SHA-256 of every state and question) |
| `typed` | typed-decisions test split: 400 cases, 2,000 decisions | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) `all/test`; compared with [Antz AI's per-question results](https://github.com/pavanjava/jev_and_laya_benchmarking) (pinned commit), paired bootstrap 95% CIs |
| `laya` | Laya's own application battery: 10 tasks × 400 | Laya's `research/scripts/bench_apps.py`, cloned at the pinned commit and run unchanged; Laya's own committed results are read from the same commit |
| `youtube` (opt-in: `--suites youtube`) | the Chrome extension's questions on 800 trending YouTube videos (100 from each of 8 categories, the creator's category as the label, split dev/test), plus the request Quietly sends Jev | astune/text_info_trending_youtube_videos_2019-04-15_to_2020-04-15 (MIT) at a pinned revision, checked against `manifests/youtube_items_manifest.json`; the kinds come from `extension/src/kinds.json` |

No dataset text is committed here: `items.py` rebuilds it, and `results/` holds only item ids and
probabilities. Metrics are the same code that produced the published numbers: accuracy, expected
calibration error (10 bins, top-option confidence) and Jensen-Shannon distance from ChaosNLI's human label
distribution; Laya's battery uses Laya's own `metrics()`.

## Committed results

| model | general | typed | laya | how it was run |
|---|---|---|---|---|
| opendecider-nano | ✅ | ✅ | ✅ | the released model through this package (identical on Apple Silicon and NVIDIA) |
| opendecider-small | ✅ | ✅ | ✅ | the released model through this package, NVIDIA L40S |
| opendecider-nano-onnx-q8, -q8f16, -fp16 | ✅ | ✅ | ✅ | @opendecider/web's builds, `onnx:<file>` in native ONNX Runtime (CPU), plus the guard sets (fp16: the attack questions) |
| jev | ✅ | ✅ | ✅ | TypeSafe's own API, Jev 1.13, 2026-09-26/27 |
| microsoft-decision-1 | ✅ | ✅ | ✅ | Microsoft-Decision-1 through our own Microsoft Foundry deployment (East US), 2026-10-10 |
| laya, laya-td | ✅ | Antz AI's run | Laya's committed run | `pip install laya` (0.3.x) for `general` |
| clm-8b | ✅ | ✅ | ✅ | Contrastive-LM/CLM-v0.1-8B with its own engine through the official vLLM pooling server |

The `youtube` suite: opendecider-nano (PyTorch) and its fp16 build, jev (TypeSafe's own API, Jev 1.13, 2026-10-05),
laya, laya-td and laya-ml (`pip install laya`, 0.3.22).

`report.py` also prints Microsoft-Decision-1 against every other model, question for question, with paired bootstrap
95% CIs.

Frontier LLM numbers (Claude Fable 5.1, GPT-6 Astra, DeepSeek V4.1 Flash, MiniMax M3, Kimi K3) come from the
same 200 general decisions in [jev-frontier-bench](https://github.com/manjunathshiva/jev-frontier-bench).

## The prompt guard benchmark

`guard.py` measures `opendecider.guard` against Laya's guard on three public datasets, each pinned to a revision:
[deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections),
[jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) and
[xTRam1/safe-guard-prompt-injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) (test splits:
2,438 prompts; prompts cut to 3,000 characters).

```bash
python benchmarks/guard.py run --model opendecider-small-td               # test split, every setup
python benchmarks/guard.py run --model opendecider-small-td --split train --setups attack
python benchmarks/guard.py run --model laya-en                            # pip install laya
# a quantised build, through Ollama (GGUF) or natively (MLX, pip install "opendecider[mlx]"): train, then test
python benchmarks/guard.py run --model "ollama:hf.co/manjunathshiva/opendecider-small-td-GGUF:Q4_K_M" \
    --name opendecider-small-td-gguf-q4 --split train --setups attack
python benchmarks/guard.py run --model "ollama:hf.co/manjunathshiva/opendecider-small-td-GGUF:Q4_K_M" \
    --name opendecider-small-td-gguf-q4 --setups attack
python benchmarks/guard.py report                                          # every table, from the committed results
```

| setup | questions | rule |
|---|---|---|
| `single` | Laya's deepset question, one yes/no | flagged at 0.5 |
| `preset` | Laya's whole guard preset (`laya.guard_questions()`, copied verbatim) | LayaGuardrail's own rule at 0.5 |
| `attack` | the preset's two attack questions, as `opendecider.guard` asks them | each model's train-split threshold |

For `attack`, each model's threshold comes from the train splits (a seeded sample of up to 1,000 prompts per dataset)
by a rule fixed before any test result was seen: the threshold that maximises balanced accuracy averaged over the
three datasets. It is then applied once to the test splits; `report` prints the test result at that threshold and at
0.5. Results: `results/guard/<model>.jsonl` (test) and `results/guard/train/<model>.jsonl`, with item ids and
probabilities only; `*.latency.json` holds the time per preset request on the machine that ran it.

## Rules we held to

* No benchmark dataset above, or its family, is in OpenDecider's training data; every training pool was checked
  for text overlap with all test sets (0 overlaps).
* opendecider-nano was fine-tuned on the typed-decisions **train** split, like Laya's typed-decisions checkpoint;
  the test split was never used for training or model selection.
