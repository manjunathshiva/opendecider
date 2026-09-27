"""Question handling, without downloading a model."""
import pytest

from opendecider import Choice, Noul, OpenDecider, Score
from opendecider.questions import answer, as_dict, options


class Uniform:
    def decide_many(self, items):
        return [{k: 1 / len(o) for k in o} for _, _, o in items]


def test_options_match_training_format():
    assert options(as_dict(Choice("Which team?", {"billing": "charges", "tech": None}))) == \
        {"billing": "charges", "tech": None}
    assert options(as_dict(Score("How urgent?", ["low", "high"]))) == {"0": "low", "1": "high"}
    assert options(as_dict(Noul("Spam?"))) == {"yes": "Yes", "no": "No"}
    assert options(as_dict(Noul("Spam?", {"true": "spam", "false": "wanted"}))) == {"yes": "spam", "no": "wanted"}


def test_choice_list_criteria():
    assert as_dict(Choice("Which?", ["a", "b"]))["criteria"] == {"a": None, "b": None}


def test_answers_are_typed():
    a = answer(as_dict(Noul("Spam?")), {"yes": 0.8, "no": 0.2})
    assert a["type"] == "noul" and a["noul"] == 0.8 and a["probabilities"] == {"true": 0.8, "false": pytest.approx(0.2)}
    s = answer(as_dict(Score("Urgent?", ["a", "b", "c"])), {"0": 0.1, "1": 0.2, "2": 0.7})
    assert s["score"] == 2 and s["expected"] == pytest.approx(1.6)
    c = answer(as_dict(Choice("Team?", ["x", "y"])), {"x": 0.3, "y": 0.7})
    assert c["choice"] == "y" and c["confidence"] == 0.7


def test_system_one_shape():
    r = OpenDecider(Uniform(), {"name": "test"}).system_one(
        {"text": "hi"}, {"q1": Noul("Is it spam?"), "q2": {"type": "choice", "instructions": "Team?", "criteria": ["a", "b", "c"]}})
    assert set(r["answers"]) == {"q1", "q2"} and r["answers"]["q2"]["probabilities"]["a"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad", [
    {"type": "vote", "instructions": "x"},
    {"type": "choice", "instructions": "x", "criteria": {"only": None}},
    {"type": "score", "instructions": "x", "criteria": "high"},
    {"type": "noul", "instructions": ""},
])
def test_bad_questions_rejected(bad):
    with pytest.raises(ValueError):
        as_dict(bad)
