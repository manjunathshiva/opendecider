"""Typed questions (choice / score / noul) and typed answers.

A question is a dict, or one of the small helper classes below:

    {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "tech": "Bugs"}}
    {"type": "score",  "instructions": "How urgent?", "criteria": ["not urgent", "soon", "now"]}
    {"type": "noul",   "instructions": "Is this spam?"}                       # yes/no
    {"type": "noul",   "instructions": "Is this spam?", "criteria": {"true": "spam", "false": "wanted mail"}}

Every model sees a question as named options; this module turns the typed form into those
options exactly as OpenDecider's benchmarks did, and turns option probabilities back into a
typed answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TYPES = ("choice", "score", "noul")


@dataclass
class Choice:
    instructions: str
    criteria: dict | list

    def to_dict(self):
        crit = {c: None for c in self.criteria} if isinstance(self.criteria, list) else dict(self.criteria)
        return {"type": "choice", "instructions": self.instructions, "criteria": crit}


@dataclass
class Score:
    instructions: str
    criteria: list          # ordered levels, lowest first

    def to_dict(self):
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass
class Noul:
    instructions: str
    criteria: dict = field(default_factory=dict)   # optional {"true": ..., "false": ...} descriptions

    def to_dict(self):
        return {"type": "noul", "instructions": self.instructions, "criteria": dict(self.criteria)}


def as_dict(q: Any) -> dict:
    q = q.to_dict() if hasattr(q, "to_dict") else dict(q)
    if q.get("type") not in TYPES:
        raise ValueError(f"question type must be one of {TYPES}, got {q.get('type')!r}")
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        raise ValueError("question needs non-empty 'instructions'")
    if q["type"] == "choice":
        crit = q.get("criteria")
        if isinstance(crit, list):
            crit = {c: None for c in crit}
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError("choice question needs 'criteria' with at least 2 options")
        q["criteria"] = crit
    elif q["type"] == "score":
        if not isinstance(q.get("criteria"), list) or len(q["criteria"]) < 2:
            raise ValueError("score question needs 'criteria' as an ordered list of at least 2 levels")
    return q


def options(q: dict) -> dict:
    """Typed question -> {option name: description or None}, as the models were trained and evaluated."""
    if q["type"] == "choice":
        return dict(q["criteria"])
    if q["type"] == "score":
        return {str(i): d for i, d in enumerate(q["criteria"])}
    crit = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
    return {"yes": crit.get("true", "Yes"), "no": crit.get("false", "No")}


def answer(q: dict, probs: dict) -> dict:
    """Option probabilities -> typed answer."""
    if q["type"] == "noul":
        p = float(probs["yes"])
        return {"type": "noul", "noul": p, "probabilities": {"true": p, "false": 1 - p}, "confidence": max(p, 1 - p)}
    probs = {k: float(v) for k, v in probs.items()}
    top = max(probs, key=probs.get)
    if q["type"] == "score":
        return {"type": "score", "score": int(top), "expected": sum(int(k) * v for k, v in probs.items()),
                "legend": {str(i): lvl for i, lvl in enumerate(q["criteria"])},
                "probabilities": probs, "confidence": probs[top]}
    return {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}
