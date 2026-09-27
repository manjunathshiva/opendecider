"""OpenDecider: small open decision models (System One) with typed questions.

    from opendecider import load, Choice, Score, Noul

    model = load("manjunathshiva/opendecider-nano")        # or "manjunathshiva/opendecider-small"
    r = model.system_one(
        state={"subject": "Refund?", "body": "I was charged twice for order 1182."},
        questions={
            "team": Choice("Which team should handle this?", {"billing": "charges, refunds", "tech": "bugs"}),
            "urgent": Noul("Does this need a reply today?"),
            "anger": Score("How upset is the customer?", ["calm", "annoyed", "angry"]),
        })
    r["answers"]["team"]["choice"], r["answers"]["team"]["probabilities"]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .questions import Choice, Noul, Score, answer, as_dict, options

__version__ = "0.1.2"
__all__ = ["load", "OpenDecider", "Choice", "Score", "Noul", "__version__"]


def default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenDecider:
    """A loaded model. `system_one(state, questions)` answers typed questions about one state."""

    def __init__(self, impl, meta: dict):
        self.impl, self.meta = impl, meta

    def system_one(self, state, questions: dict) -> dict:
        if not questions:
            raise ValueError("questions must not be empty")
        qs = {k: as_dict(q) for k, q in questions.items()}
        t0 = time.perf_counter()
        probs = self.impl.decide_many([(state, q["instructions"], options(q)) for q in qs.values()])
        return {"model": self.meta.get("name"),
                "answers": {k: answer(q, p) for (k, q), p in zip(qs.items(), probs)},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def load(name_or_path: str = "manjunathshiva/opendecider-nano", device: str | None = None,
         revision: str | None = None) -> OpenDecider:
    """Load a model from the Hugging Face Hub or a local folder (anything with opendecider.json)."""
    path = Path(name_or_path).expanduser()
    if not (path / "opendecider.json").exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(name_or_path, revision=revision))
    meta = json.loads((path / "opendecider.json").read_text())
    if meta["kind"] != "small-mlx":   # MLX builds need neither torch nor a device choice
        device = device or default_device()
    if meta["kind"] == "nano":
        from .nano import NanoModel
        impl = NanoModel(str(path), device, meta.get("max_len", 2048))
    elif meta["kind"] == "small":
        from .small import SmallModel
        impl = SmallModel(str(path), meta["base_model"], device,
                          meta.get("device_map") if device == "cuda" else None)
    elif meta["kind"] == "small-mlx":   # merged + quantised build for Apple Silicon (pip install "opendecider[mlx]")
        from .mlx_small import MLXSmallModel
        impl = MLXSmallModel(str(path), meta.get("mlx_base"))
    else:
        raise ValueError(f"unknown model kind {meta['kind']!r}")
    return OpenDecider(impl, meta)
