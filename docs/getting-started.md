# Get started

## Install

Python 3.10 or newer, on Linux, Windows or macOS.

```bash
pip install opendecider              # opendecider-nano
pip install "opendecider[small]"     # adds peft for opendecider-small, -small-td, -medium-td and -large-td
pip install "opendecider[mlx]"       # Apple Silicon: the MLX 8-bit / 4-bit builds of opendecider-small
pip install "opendecider[serve]"     # the HTTP server (Jev-compatible /v1/systemone)
```

For a clean setup with the right PyTorch for your hardware:

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

!!! note "Google Colab"
    Run `pip uninstall -y torchao` before loading small or small-td. Colab preinstalls torchao 0.10, which recent peft
    refuses to load LoRA adapters next to ("Found an incompatible version of torchao"). OpenDecider does not use
    torchao.

## Your first decision

```python
from opendecider import load

model = load("manjunathshiva/opendecider-nano")   # 0.8 GB, downloaded on first use

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

The **state** can be plain text or any JSON-serialisable object: a ticket with subject, body and customer fields, a log
record, an agent's tool-call trace. The **questions** are named, and each one gets a typed answer.

## Three question types

Write questions as dicts (as above) or with the helper classes:

```python
from opendecider import Choice, Score, Noul

Choice("Which team?", {"billing": "charges, refunds", "technical": "bugs"})   # pick one; descriptions optional
Choice("Which intent?", ["refund", "replacement", "information"])             # a plain list of labels
Score("How urgent?", ["not urgent", "soon", "blocking"])                      # ordered levels, lowest first
Noul("Is this spam?")                                                          # yes / no
Noul("Is this spam?", {"true": "unsolicited marketing", "false": "mail the user wants"})
```

Every answer carries `probabilities` over its options and a `confidence` (the probability of the top answer):

```python
{"type": "choice", "choice": "billing", "probabilities": {"billing": 0.927, ...}, "confidence": 0.927}
{"type": "score",  "score": 2, "expected": 1.51, "probabilities": {"0": 0.091, "1": 0.305, "2": 0.604}, "confidence": 0.604}
{"type": "noul",   "noul": 0.922, "probabilities": {"true": 0.922, "false": 0.078}, "confidence": 0.922}
```

!!! tip "Descriptions help"
    Very terse or cryptic option labels are harder for every model. Give options a short description when you can.

## Many states at once

`system_one_batch` asks the same questions about a list of states in one call. opendecider-nano runs them as one
padded batch, much faster than one call per state:

```python
results = model.system_one_batch(["The app crashes on login.", "How do I download my invoices?"], questions)
```

## Devices, offline use and memory

- **Device:** CUDA, then MPS, then CPU, chosen automatically. Override with `load(..., device="cpu")`.
- **Offline or air-gapped:** download a model folder once
  (`huggingface-cli download manjunathshiva/opendecider-nano --local-dir ./nano`), then `load("./nano")`.
- **CPU only:** nano runs fine on CPU for batch jobs. small needs about 17 GB of RAM in fp32 and is slow on CPU.
- **Memory:** nano 2.0 GiB, small 8.9 GiB of GPU or unified memory (measured on a 16 GB Mac mini M4).
- **Speed on your machine:** `python -m opendecider.bench_speed manjunathshiva/opendecider-nano`.

## Next

- [Choose a model](models.md): nano, small, small-td, medium-td, large-td, and the MLX and GGUF builds.
- [Serve it](guides/serve.md): the same answers over HTTP, compatible with TypeSafe Jev's API.
- [Examples and notebook](examples.md): runnable scripts for triage, guardrails and automation.
