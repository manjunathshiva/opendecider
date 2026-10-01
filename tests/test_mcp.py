"""MCP server: the tools, their answers and errors, through a real in-process MCP client, with a fake model."""
import asyncio
import sys
import textwrap

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from opendecider import OpenDecider  # noqa: E402
from opendecider.mcp_server import MAX_OPTIONS, Decider, build_server  # noqa: E402


class Fake:
    """The first option gets 0.7, the rest share 0.3."""

    def __init__(self):
        self.calls = 0

    def decide_many(self, items, info=None):
        self.calls += 1
        out = []
        for state, _, opts in items:
            names = list(opts)
            rest = 0.3 / (len(names) - 1)
            out.append({k: 0.7 if i == 0 else rest for i, k in enumerate(names)})
            if info is not None:
                info.append({"input_tokens": 10, "truncated": state == "very long"})
        return out


def setup():
    fake, loads = Fake(), []

    def loader(name, **kw):
        loads.append(name)
        return OpenDecider(fake, {"name": "opendecider-test", "kind": "nano"})

    return build_server(Decider("test-model", loader=loader)), fake, loads


def call(server, tool, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(tool, args)
    return asyncio.run(go())


def test_tools_are_listed_read_only():
    server, _, loads = setup()

    async def go():
        async with Client(server) as c:
            return (await c.list_tools()).tools
    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {"decide", "choose", "yes_no", "score", "decide_batch", "status"}
    assert all(t.annotations.read_only_hint and t.annotations.idempotent_hint for t in tools.values())
    assert loads == []   # listing tools does not load the model


def test_decide_answers_every_question_type():
    server, fake, loads = setup()
    r = call(server, "decide", {"state": {"subject": "Refund?", "body": "Charged twice."}, "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs"}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
        "churn": {"type": "noul", "instructions": "Will they cancel?"}}})
    assert not r.is_error
    a = r.structured_content["answers"]
    assert a["team"] == {"choice": "billing", "probabilities": {"billing": 0.7, "tech": 0.3}, "confidence": 0.7}
    assert a["urgency"]["level"] == 0 and a["urgency"]["label"] == "low"
    assert a["urgency"]["probabilities"]["high"] == 0.3
    assert a["churn"]["answer"] == "yes" and a["churn"]["probability_yes"] == 0.7
    assert loads == ["test-model"] and fake.calls == 1


def test_shortcuts():
    server, _, loads = setup()
    r = call(server, "choose", {"state": "Charged twice", "question": "Which team?", "options": ["billing", "tech"]})
    assert r.structured_content["choice"] == "billing"
    r = call(server, "choose", {"state": "x", "question": "Which?", "options": {"a": "first", "b": "second"}})
    assert r.structured_content["probabilities"] == {"a": 0.7, "b": 0.3}
    r = call(server, "yes_no", {"state": "Buy now!!!", "question": "Is this spam?"})
    assert r.structured_content == {"answer": "yes", "probability_yes": 0.7, "confidence": 0.7}
    r = call(server, "score", {"state": "x", "question": "How bad?", "levels": ["mild", "severe"]})
    assert r.structured_content["label"] == "mild" and r.structured_content["expected_level"] == 0.3
    assert loads == ["test-model"]   # loaded once, on the first call


def test_truncation_is_reported():
    server, _, _ = setup()
    r = call(server, "yes_no", {"state": "very long", "question": "Is it?"})
    assert r.structured_content["truncated"] is True


def test_invalid_input_is_a_tool_error_with_a_message():
    server, _, loads = setup()
    r = call(server, "decide", {"state": "x", "questions": {"q": {"type": "pick", "instructions": "?"}}})
    assert r.is_error and "question type must be one of" in r.content[0].text
    assert loads == []   # rejected before the model loads: a bad first call never triggers a download
    too_many = [f"o{i}" for i in range(MAX_OPTIONS + 1)]
    r = call(server, "choose", {"state": "x", "question": "Which?", "options": too_many})
    assert r.is_error and f"at most {MAX_OPTIONS} options" in r.content[0].text
    r = call(server, "yes_no", {"state": "x" * 200_001, "question": "?"})
    assert r.is_error and "longer than" in r.content[0].text


def test_duplicate_score_levels_are_rejected():
    server, _, _ = setup()
    r = call(server, "score", {"state": "x", "question": "How bad?", "levels": ["low", "low"]})
    assert r.is_error and "score levels must be distinct" in r.content[0].text


def test_a_model_that_cannot_load_says_why_and_is_retried():
    attempts = []

    def loader(name, **kw):
        attempts.append(name)
        raise OSError("no such model on the Hub")

    server = build_server(Decider("typo/model", loader=loader))
    for _ in range(2):
        r = call(server, "yes_no", {"state": "x", "question": "?"})
        assert r.is_error
        assert "could not load model 'typo/model': OSError: no such model on the Hub" in r.content[0].text
    assert attempts == ["typo/model"]   # within LOAD_RETRY_S the second call fails fast with the same reason


NOISY_SERVER = textwrap.dedent("""
    import os
    from opendecider import OpenDecider
    from opendecider.mcp_server import Decider, build_server

    class Noisy:
        def decide_many(self, items, info=None):
            print("library output during inference")      # Python-level stdout
            os.write(1, b"native output on fd 1\\n")       # C-level stdout
            info.extend({"input_tokens": 1, "truncated": False} for _ in items)
            return [{k: 0.7 if i == 0 else 0.3 for i, k in enumerate(o)} for _, _, o in items]

    def loader(name, **kw):
        print("library output while loading")
        return OpenDecider(Noisy(), {"name": "noisy", "kind": "nano"})

    build_server(Decider("noisy", loader=loader)).run("stdio")
""")


def test_library_output_never_reaches_the_protocol(tmp_path, caplog):
    from mcp import StdioServerParameters
    script = tmp_path / "noisy_server.py"
    script.write_text(NOISY_SERVER)

    async def go():
        async with Client(StdioServerParameters(command=sys.executable, args=[str(script)])) as c:
            return [await c.call_tool("yes_no", {"state": "x", "question": "?"}) for _ in range(2)]
    results = asyncio.run(go())
    assert all(not r.is_error and r.structured_content["answer"] == "yes" for r in results)
    assert "Invalid JSON" not in caplog.text and "library output" not in caplog.text   # nothing leaked onto stdout


def test_decide_batch_answers_each_state_in_one_model_call():
    server, fake, _ = setup()
    qs = {"team": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]},
          "churn": {"type": "noul", "instructions": "Will they leave?"}}
    r = call(server, "decide_batch", {"states": ["one", {"ticket": "two"}, "very long"], "questions": qs})
    out = r.structured_content
    assert out["model"] == "opendecider-test" and len(out["results"]) == 3
    assert out["results"][0]["answers"]["team"]["choice"] == "billing"
    assert out["results"][2]["answers"]["team"]["truncated"] is True and "warnings" in out["results"][2]
    assert fake.calls == 1   # every state and question in one batch
    bad = call(server, "decide_batch", {"states": [], "questions": qs})
    assert bad.is_error and "non-empty list" in bad.content[0].text
    too_many = call(server, "decide_batch", {"states": ["x"] * 257, "questions": qs})
    assert too_many.is_error and "at most 256 states" in too_many.content[0].text
    many_qs = {f"q{i}": {"type": "noul", "instructions": f"question {i}?"} for i in range(5)}
    too_much = call(server, "decide_batch", {"states": ["x"] * 256, "questions": many_qs})   # 1,280 > 1,024
    assert too_much.is_error and "at most 1024 questions in all" in too_much.content[0].text


def test_status_reports_without_loading_the_model():
    server, _, loads = setup()
    st = call(server, "status", {}).structured_content
    assert st["model"] == "test-model" and st["loaded"] is False and loads == []
    assert st["limits"] == {"questions": 64, "options": 256, "state_chars": 200_000, "batch_states": 256,
                            "batch_items": 1024}
    call(server, "yes_no", {"state": "s", "question": "q?"})
    st = call(server, "status", {}).structured_content
    assert st["loaded"] is True and st["kind"] == "nano" and "version" in st
