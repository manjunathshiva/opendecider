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

__version__ = "0.5.0"
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
        import inspect
        try:
            self._takes_info = "info" in inspect.signature(impl.decide_many).parameters
        except (TypeError, ValueError):
            self._takes_info = False

    @property
    def name(self) -> str:
        return self.meta.get("name") or "opendecider"

    @staticmethod
    def prepare(questions: dict) -> dict:
        """Validate typed questions and normalise them to dicts (raises ValueError with a readable message)."""
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object of named questions")
        out = {}
        for k, q in questions.items():
            try:
                out[str(k)] = as_dict(q)
            except ValueError as e:
                raise ValueError(f"question {k!r}: {e}") from None
        return out

    def decide(self, items: list[tuple]) -> tuple[list[dict], list[dict]]:
        """[(state, instructions, options)] -> (option probabilities, per-item {input_tokens, truncated})."""
        if self._takes_info:
            info: list = []
            probs = self.impl.decide_many(items, info=info)
        else:
            probs, info = self.impl.decide_many(items), [{} for _ in items]
        return probs, info

    @staticmethod
    def items(state, qs: dict) -> list[tuple]:
        return [(state, q["instructions"], options(q)) for q in qs.values()]

    def assemble(self, qs: dict, probs: list[dict], info: list[dict]) -> dict:
        """Typed answers in the System One response shape: {model, answers, usage[, warnings]}."""
        answers, warnings = {}, []
        for (k, q), p, inf in zip(qs.items(), probs, info):
            answers[k] = answer(q, p)
            if inf.get("truncated"):
                answers[k]["truncated"] = True
                warnings.append(f"question {k!r}: the state was truncated to fit the model's input length")
        out = {"model": self.name, "answers": answers,
               "usage": {"input_tokens": sum(i.get("input_tokens", 0) for i in info), "output_tokens": 0}}
        if warnings:
            out["warnings"] = warnings
        return out

    def system_one(self, state, questions: dict) -> dict:
        qs = self.prepare(questions)
        t0 = time.perf_counter()
        out = self.assemble(qs, *self.decide(self.items(state, qs)))
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def system_one_batch(self, states: list, questions: dict) -> list[dict]:
        """The same questions about many states, in one call (one padded batch for nano)."""
        qs = self.prepare(questions)
        n = len(qs)
        probs, info = self.decide([it for st in states for it in self.items(st, qs)])
        return [self.assemble(qs, probs[i * n:(i + 1) * n], info[i * n:(i + 1) * n]) for i in range(len(states))]


def load(name_or_path: str = "manjunathshiva/opendecider-nano", device: str | None = None,
         revision: str | None = None, dtype: str | None = None, base_url: str | None = None,
         api_key: str | None = None, timeout: float | None = None) -> OpenDecider:
    """Load a model from the Hub or a local folder (anything with opendecider.json).
    `dtype` (nano only): "float32" (default, as evaluated) or "bfloat16" (faster on CPUs with bf16 units and on GPUs).
    "lmstudio:<model>", "ollama:<model>" or "openai:<model>" (with `base_url`) uses a model served by LM Studio,
    Ollama or any OpenAI-compatible server that returns token log-probabilities (see opendecider.remote).
    An http(s) URL uses the model behind `opendecider serve` at that address, with no local model.
    `api_key` and `timeout` (seconds per request) apply to served models; the key defaults to
    OPENDECIDER_REMOTE_API_KEY."""
    from . import remote
    http = {k: v for k, v in (("api_key", api_key), ("timeout", timeout)) if v is not None}
    if remote.is_url(name_or_path):   # opendecider serve, by URL
        impl = remote.ServedModel(name_or_path, **http)
        return OpenDecider(impl, {"name": impl.name, "kind": "served", "served_kind": impl.kind,
                                  "base_url": impl.base_url})
    target = remote.parse(name_or_path, base_url)
    if target:   # a model served by LM Studio / Ollama / any OpenAI-compatible server with logprobs
        impl = remote.RemoteModel(target[0], target[1], **http)
        return OpenDecider(impl, {"name": target[0], "kind": "remote", "base_url": target[1]})
    path = Path(name_or_path).expanduser()
    if not (path / "opendecider.json").exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(name_or_path, revision=revision))
    meta = json.loads((path / "opendecider.json").read_text())
    if meta["kind"] != "small-mlx":   # MLX builds need neither torch nor a device choice
        device = device or default_device()
    if meta["kind"] == "nano":
        from .nano import NanoModel
        impl = NanoModel(str(path), device, meta.get("max_len", 2048), dtype or "float32")
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
