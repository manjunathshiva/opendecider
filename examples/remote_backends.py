"""The same decision through a model server: LM Studio, Ollama or vLLM.

    pip install "opendecider>=0.2.1"      # no torch needed: the server runs the model

    # LM Studio: load the Q8_0 file of opendecider-small-GGUF and start its server
    python examples/remote_backends.py lmstudio:opendecider-small
    # Ollama: ollama pull hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0
    python examples/remote_backends.py ollama:hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0
    # vLLM: the base model with the LoRA adapter (see the README, "Run it in LM Studio or Ollama")
    python examples/remote_backends.py openai:opendecider-small --base-url http://localhost:8000/v1

The server only runs the model. OpenDecider sends the prompt the model was trained on and reads the option
probabilities from the server's token log-probabilities, so answers match the PyTorch model on about 99% of questions.
"""
import argparse

from opendecider import Choice, Noul, Score, load

ap = argparse.ArgumentParser()
ap.add_argument("model", help="lmstudio:<name>, ollama:<name> or openai:<name>")
ap.add_argument("--base-url", help="server URL (default: LM Studio's or Ollama's local address)")
a = ap.parse_args()

model = load(a.model, base_url=a.base_url)
r = model.system_one(
    {"invoice_id": "INV-2291", "vendor": "Acme Supplies", "amount": 4820.00, "currency": "USD",
     "po_number": None, "due": "2026-09-15", "note": "Second reminder, now 12 days overdue."},
    {"action": Choice("What should accounts payable do with this invoice?",
                      {"approve": "pay it", "hold": "hold for a missing purchase order", "reject": "not a valid invoice"}),
     "risk": Score("How risky is paying this invoice?", ["low", "medium", "high"]),
     "needs_review": Noul("Should a human review this before payment?")})

print(f"{a.model} at {model.meta['base_url']}: {r['latency_ms']:.0f} ms for {len(r['answers'])} questions")
for name, ans in r["answers"].items():
    print(f"  {name:13s} {ans['type']:6s} {({k: round(v, 3) for k, v in ans['probabilities'].items()})}")
