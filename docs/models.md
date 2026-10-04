# Choose a model

Every model answers the same typed questions with the same API; they differ in speed, memory and what they are best
at. All are Apache-2.0 and on [Hugging Face](https://huggingface.co/collections/manjunathshiva/opendecider-6ab8c838909092518d50a9ea).

## Which one?

| If you need… | Use |
|---|---|
| many decisions per second, on CPU or any GPU | **opendecider-nano** |
| the best accuracy on decisions unlike the training data, on a 16 GB Mac or one GPU | **opendecider-small** |
| business workflows (triage, invoices, security alerts, agent traces) on a 16 GB Mac or one GPU | **opendecider-small-td** |
| the most accurate model you can host yourself (NVIDIA, several GPUs) | **opendecider-medium-td** |
| probabilities you can threshold on: the best calibration and agreement with people (NVIDIA, several GPUs) | **opendecider-large-td** |
| a Mac with MLX | **opendecider-small-mlx-8bit** (or -4bit with little memory) |
| LM Studio or Ollama | **opendecider-small-GGUF** / **opendecider-small-td-GGUF** |

## All checkpoints

| model | backbone | params | memory | use it for |
|---|---|---|---|---|
| [`opendecider-nano`](https://huggingface.co/manjunathshiva/opendecider-nano) | Ettin-encoder-400m | ~400M | 2.0 GiB | speed: 17–18 ms per question, ~9 ms batched; typed business decisions |
| [`opendecider-small`](https://huggingface.co/manjunathshiva/opendecider-small) | Qwen3-4B-Instruct-2507 + LoRA | 4B | 8.9 GiB, tested on a 16 GB Mac mini | accuracy and calibration on decisions it has never seen |
| [`opendecider-small-td`](https://huggingface.co/manjunathshiva/opendecider-small-td) | Qwen3-4B-Instruct-2507 + LoRA | 4B | 8.9 GiB | business workflows like typed-decisions' (triage, invoices, security alerts, agent traces): 0.792 |
| [`opendecider-medium-td`](https://huggingface.co/manjunathshiva/opendecider-medium-td) | Qwen3-30B-A3B-Instruct-2507 + LoRA | 30B (3B active) | 61 GB bf16, across several GPUs (tested on 4× L40S) | the most accurate self-hostable model on general decisions (0.765); NVIDIA only |
| [`opendecider-large-td`](https://huggingface.co/manjunathshiva/opendecider-large-td) | Qwen3-Next-80B-A3B-Instruct + LoRA | 80B (3B active) | 160 GB bf16, across several GPUs (tested on 4× L40S) | the best calibration and agreement with people; NVIDIA only |
| [`opendecider-small-mlx-8bit`](https://huggingface.co/manjunathshiva/opendecider-small-mlx-8bit) | opendecider-small, MLX 8-bit | 4B | 4.5 GB | Macs: same answers as full precision on 1,955 of 2,000 typed-decisions questions, 66 ms per question |
| [`opendecider-small-mlx-4bit`](https://huggingface.co/manjunathshiva/opendecider-small-mlx-4bit) | opendecider-small, MLX 4-bit | 4B | 2.6 GB | Macs with little memory; about 2 points lower on typed-decisions (0.651) |
| [`opendecider-small-GGUF`](https://huggingface.co/manjunathshiva/opendecider-small-GGUF) | opendecider-small, GGUF Q8_0 / Q4_K_M | 4B | 4.3 / 2.5 GB | LM Studio and Ollama: typed-decisions 0.669 at Q8_0 (full precision 0.671) |
| [`opendecider-small-td-GGUF`](https://huggingface.co/manjunathshiva/opendecider-small-td-GGUF) | opendecider-small-td, GGUF Q8_0 / Q4_K_M | 4B | 4.3 / 2.5 GB | LM Studio and Ollama, business workflows: 0.794 at Q8_0 (full precision 0.792) |

Load any of them by name: `load("manjunathshiva/opendecider-small-td")`. The Qwen-based models need
`pip install "opendecider[small]"`, the MLX builds `pip install "opendecider[mlx]"`, and the GGUF builds run in an app
(see [LM Studio, Ollama and vLLM](guides/model-servers.md)).

## Hardware

| hardware | nano | small / small-td | medium-td | large-td |
|---|---|---|---|---|
| CPU only | ✅ 0.1–0.7 s per question | slow (~17 GB RAM in fp32) | – | – |
| 16 GB Mac (M4) | ✅ 28 ms | ✅ 280 ms (8.9 GiB of the 11.8 GiB GPU budget) | – | – |
| 64 GB Mac (M4 Max) | ✅ 18 ms | ✅ 141 ms; MLX 8-bit 66 ms | – | – |
| one NVIDIA GPU | ✅ 16 ms on an L40S | ✅ 38 ms on an L40S (12 GB or more) | – | – |
| several NVIDIA GPUs | | | ✅ 214 ms on 4× L40S (~64 GB in total) | ✅ 440 ms on 4× L40S (~170 GB in total) |

Latency is one question, median. Answers are identical to four decimals across the tested Macs and NVIDIA machines.
Models larger than one GPU are spread across all visible GPUs automatically. `pip install flash-linear-attention` speeds
up large-td.

## How they work

- **opendecider-nano:** Ettin-encoder-400m (bidirectional, fully fine-tuned) reads
  `question: …, [MASK] option 1, [MASK] option 2, …, input: <state>`. The hidden state at each `[MASK]` goes through a
  small MLP to one logit, then a softmax across that question's options. The answer space is defined at request time,
  so new schemas need no retraining, and a 78-option question still costs one forward pass.
- **opendecider-small and small-td:** Qwen3-4B-Instruct-2507 with a LoRA adapter (r = 16, all linear projections). The
  options are lettered, and one forward pass gives the probability of each letter as the next token. Above 26 options
  it scores each option name's log-probability after the shared prompt.
- **opendecider-medium-td and large-td:** the same design on Qwen3-30B-A3B-Instruct-2507 and Qwen3-Next-80B-A3B-Instruct
  (mixtures of experts, 3B active). The adapter covers the attention projections; the experts are frozen.

**Training.** Two openly licensed teachers, Qwen3-235B-A22B-Instruct-2507 (Apache-2.0) and DeepSeek V4.1 Flash (MIT),
scored every training question through token log-probabilities. Each teacher was temperature-scaled on held-out gold
labels before the two were averaged, so the students learn calibrated distributions, not hard labels. nano, small-td,
medium-td and large-td then had a short fine-tune on the typed-decisions train split. No benchmark dataset, or its
family, is in the training data, and no outputs of Claude or GPT models were used. Training-data licences are in
[NOTICE](https://github.com/manjunathshiva/opendecider/blob/main/NOTICE).
