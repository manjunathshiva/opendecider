# LM Studio, Ollama and vLLM

The 4B models (small and small-td) can run in an app instead of PyTorch. The app runs the model; the `opendecider`
package builds the prompt the model was trained on and reads the option probabilities from the server's token
log-probabilities. The client never runs the model, so it needs no GPU and little memory (PyTorch is installed with the
package but not used).

```bash
pip install "opendecider[serve]>=0.2.1"      # or plain opendecider>=0.2.1 if you only need load()
```

| engine | build | agreement with the PyTorch model on typed-decisions |
|---|---|---|
| LM Studio, Ollama (llama.cpp) | [small-GGUF](https://huggingface.co/manjunathshiva/opendecider-small-GGUF), [small-td-GGUF](https://huggingface.co/manjunathshiva/opendecider-small-td-GGUF), Q8_0 | same top answer on 1,975 and 1,972 of 2,000; accuracy 0.669 vs 0.671 and 0.794 vs 0.792 |
| LM Studio, Ollama | Q4_K_M | about 93%, for machines with little memory |
| vLLM (NVIDIA) | the base model with the LoRA adapter, no merge | same top answer on 1,963 and 1,969 of 2,000; 0.6735 vs 0.6715 and 0.7945 vs 0.792 |

## LM Studio

```bash
lms get https://huggingface.co/manjunathshiva/opendecider-small-GGUF --select   # choose Q8_0 (or search "opendecider" in the app)
lms load opendecider-small@q8_0 --identifier opendecider-small
lms server start
```

```python
from opendecider import load

model = load("lmstudio:opendecider-small")
print(model.system_one("I was charged twice. Please refund the extra payment.",
                       {"team": {"type": "choice", "instructions": "Which team?",
                                 "criteria": {"billing": "payments, refunds", "technical": "bugs"}}})["answers"])
```

For the [guard](agent-guardrails.md), name the model with its variant (`--identifier opendecider-small@q8_0`, then
`load("lmstudio:opendecider-small@q8_0")`): the guard takes each build's measured threshold from the name, and a
4-bit build needs its own.

Load the GGUF build in LM Studio on a Mac too: its MLX engine returns no log-probabilities. For MLX, use the
[MLX builds](../models.md) with `pip install "opendecider[mlx]"` instead.

## Ollama

```bash
ollama pull hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0
```

```python
model = load("ollama:hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0")
```

!!! note "Ollama's own `/v1/systemone`"
    Ollama 0.35 added its own decision endpoint (open to third-party models from 0.35.1). It builds a different
    prompt from the one these builds were trained on, which costs them several points. Use them through the
    `opendecider` package as above, or through `opendecider serve` (below). Builds trained on Ollama's prompt as well
    are in preparation.

## vLLM

Serve the base model with the LoRA adapter; no merge or GGUF needed.

```bash
hf download manjunathshiva/opendecider-small --local-dir opendecider-small
vllm serve Qwen/Qwen3-4B-Instruct-2507 --enable-lora --max-lora-rank 16 --max-logprobs 20 --max-model-len 4096 \
  --lora-modules opendecider-small=./opendecider-small
```

```python
model = load("openai:opendecider-small", base_url="http://localhost:8000/v1")
```

Tested with vLLM 0.30 on an NVIDIA L4, with both adapters on one server (add `opendecider-small-td=...` to
`--lora-modules`).

## Other servers

Any server with an OpenAI-compatible `/v1/chat/completions` that returns `top_logprobs` works the same way:
`load("openai:<model>", base_url="http://host:port/v1")`.

## Jev's API on top of any of them

```bash
opendecider serve --model lmstudio:opendecider-small
opendecider serve --model ollama:hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0
```

For `openai:` models, set `OPENDECIDER_REMOTE_URL` to the server's URL. See [Serve it](serve.md).

## Settings and limits

- **Server URL:** `base_url=...`, else `OPENDECIDER_REMOTE_URL`, else LM Studio's (`http://127.0.0.1:1234/v1`) or
  Ollama's (`http://127.0.0.1:11434/v1`) default address.
- **API key:** set `OPENDECIDER_REMOTE_API_KEY` if the server needs one. It is sent only to that server (never on a
  redirect), and over plain HTTP only to this machine; for another host use https, or set
  `OPENDECIDER_REMOTE_ALLOW_HTTP=1` if the network path is trusted.
- **Up to 26 options per question.** OpenAI-compatible servers return at most 20 log-probabilities, so with 21 to 26
  options the least likely letters get probabilities near zero.
