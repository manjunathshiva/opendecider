"""Routing decisions: the Decision record, on_decision hooks, the on_error policy and OpenTelemetry spans."""
import logging

import pytest

from opendecider import OpenDecider
from opendecider.tools import Decider, Decision, Router

ROUTES = {"billing": "charges, refunds", "tech": "bugs, outages"}


class Fake:
    """The option whose name appears in the state gets 0.8 (else the first option); 'boom' fails inference."""

    def decide_many(self, items, info=None):
        out = []
        for state, _, opts in items:
            if state == "boom":
                raise RuntimeError("model gone")
            names = list(opts)
            top = next((n for n in names if n in str(state)), names[0])
            out.append({n: 0.8 if n == top else 0.2 / (len(names) - 1) for n in names})
            if info is not None:
                info.append({"input_tokens": 5, "truncated": False})
        return out


def decider():
    return Decider(OpenDecider(Fake(), {"name": "opendecider-test", "kind": "nano"}))


def test_a_decision_says_what_was_taken_and_why():
    route = Router(ROUTES, "Which team?", model=decider(), fallback="human", min_confidence=0.9)
    d = route.decide("the tech stack is down")
    assert (d.route, d.reason, d.choice, d.confidence) == ("human", "low_confidence", "tech", 0.8)
    assert d.probabilities == {"billing": pytest.approx(0.2), "tech": pytest.approx(0.8)}
    assert d.model == "opendecider-test" and d.latency_ms >= 0 and d.error is None
    route.min_confidence = 0.5
    assert route.decide("refund my billing").reason == "top_choice" and route("refund my billing") == "billing"
    empty = route.decide("  ")
    assert (empty.route, empty.reason, empty.choice, empty.probabilities) == ("human", "empty_input", None, {})
    assert set(d.to_dict()) == {"route", "reason", "choice", "confidence", "probabilities", "model", "latency_ms",
                                "truncated", "error"}


def test_hooks_see_every_decision_and_never_break_routing(caplog):
    seen, also = [], []

    def broken(d):
        raise RuntimeError("hook bug")

    route = Router(ROUTES, "Which team?", model=decider(), fallback="human", on_decision=[seen.append, broken,
                                                                                          also.append])
    assert route("refund my billing") == "billing"
    assert [d.route for d in seen] == ["billing"] == [d.route for d in also]   # the broken hook stopped nothing
    assert "on_decision hook failed" in caplog.text
    single = []
    Router(ROUTES, "Which team?", model=decider(), on_decision=single.append)("tech")
    assert single[0].route == "tech"
    with pytest.raises(ValueError, match="on_decision"):
        Router(ROUTES, "Which team?", model=decider(), on_decision="not callable")


def test_on_error_fallback_takes_the_fallback_and_reports_the_error(caplog):
    seen = []
    route = Router(ROUTES, "Which team?", model=decider(), fallback="human", on_error="fallback",
                   on_decision=seen.append)
    with caplog.at_level(logging.ERROR):
        assert route("boom") == "human"
    d = seen[0]
    assert (d.route, d.reason, d.choice) == ("human", "error", None) and "model gone" in d.error
    assert "routing failed, taking the fallback route 'human'" in caplog.text
    assert route.last is d


def test_on_error_raise_still_reports_the_failure():
    seen = []
    route = Router(ROUTES, "Which team?", model=decider(), fallback="human", on_decision=seen.append)
    with pytest.raises(RuntimeError, match="model gone"):
        route("boom")
    assert (seen[0].route, seen[0].reason) == (None, "error") and "RuntimeError" in seen[0].error


def test_on_error_is_validated():
    with pytest.raises(ValueError, match='"raise" or "fallback"'):
        Router(ROUTES, "Which team?", model=decider(), fallback="human", on_error="ignore")
    with pytest.raises(ValueError, match="needs a fallback"):
        Router(ROUTES, "Which team?", model=decider(), on_error="fallback")


def test_every_framework_router_takes_the_options():
    pytest.importorskip("langchain_core")
    from opendecider.integrations.langchain import DecisionRouter
    seen = []
    route = DecisionRouter(ROUTES, "Which team?", model=decider(), state_key="t", fallback="human",
                           on_error="fallback", on_decision=seen.append)
    assert route({"t": "boom"}) == "human" and seen[0].reason == "error"


@pytest.fixture()
def spans(monkeypatch):
    """OpenDecider's spans into an in-memory exporter. The tracer is swapped directly, since frameworks loaded by other
    tests can install their own global provider, which OpenTelemetry lets be set only once. OTEL_SDK_DISABLED (set to
    keep framework telemetry off in CI) would make this provider a no-op too, so it is cleared for this test."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    otel = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opendecider import tools
    exporter = InMemorySpanExporter()
    provider = otel.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tools, "_TRACER", provider.get_tracer("opendecider"))
    return exporter


def test_each_decision_is_an_opentelemetry_span(spans):
    route = Router(ROUTES, "Which team?", model=decider(), fallback="human", min_confidence=0.5)
    route("the tech stack is down")
    by_name = {s.name: s for s in spans.get_finished_spans()}
    r, d = by_name["opendecider.route"], by_name["opendecider.decide"]
    assert d.parent.span_id == r.context.span_id   # the model call nests inside the routing decision
    assert {k: r.attributes[k] for k in ("opendecider.route", "opendecider.choice", "opendecider.reason")} == \
        {"opendecider.route": "tech", "opendecider.choice": "tech", "opendecider.reason": "top_choice"}
    assert r.attributes["opendecider.confidence"] == pytest.approx(0.8)
    assert d.attributes["opendecider.model"] == "opendecider-test" and d.attributes["opendecider.questions"] == 1


def test_a_failed_decision_is_an_error_span(spans):
    from opentelemetry.trace import StatusCode
    route = Router(ROUTES, "Which team?", model=decider(), fallback="human", on_error="fallback")
    route("boom")
    r = next(s for s in spans.get_finished_spans() if s.name == "opendecider.route")
    assert r.status.status_code == StatusCode.ERROR and r.attributes["opendecider.reason"] == "error"
    assert r.attributes["opendecider.route"] == "human"
    assert any(e.name == "exception" for e in r.events)


def test_decision_is_importable_from_every_router_module():
    assert Decision.__module__ == "opendecider.tools"


def test_a_model_server_takes_concurrent_calls_but_a_local_model_takes_turns():
    """A served model answers concurrent calls at once (the server batches them); a local torch model is one at a
    time. Two calls must meet inside the model to pass the barrier, which the lock would prevent."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from opendecider.tools import yes_no

    def meeting(thread_safe):
        gate = threading.Barrier(2, timeout=2)

        class Model(Fake):
            def decide_many(self, items, info=None):
                gate.wait()
                return super().decide_many(items, info)

        Model.thread_safe = thread_safe
        d = Decider(OpenDecider(Model(), {"name": "m", "kind": "served"}))
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(yes_no, d, "s", "q?") for _ in range(2)]
            try:
                return all(f.result()["answer"] for f in futures)
            except threading.BrokenBarrierError:
                return False

    assert meeting(thread_safe=True) is True     # served: both calls in flight together
    assert meeting(thread_safe=False) is False   # local: the second call waits its turn


def test_a_failed_load_fails_fast_then_is_retried(monkeypatch):
    from opendecider import tools
    from opendecider.tools import ModelError, yes_no
    calls, now = [], [100.0]
    monkeypatch.setattr(tools.time, "monotonic", lambda: now[0])

    def loader(name, **kw):
        calls.append(name)
        if len(calls) < 3:
            raise ConnectionError("server down")
        return OpenDecider(Fake(), {"name": "m", "kind": "nano"})

    d = Decider("http://decider:8000", loader=loader)
    for _ in range(3):   # one real attempt, then fast failures with the same message
        with pytest.raises(ModelError, match="server down"):
            yes_no(d, "s", "q?")
    assert len(calls) == 1
    now[0] += tools.LOAD_RETRY_S + 0.1
    with pytest.raises(ModelError):
        yes_no(d, "s", "q?")                    # the second real attempt, after the wait
    now[0] += tools.LOAD_RETRY_S + 0.1
    assert yes_no(d, "s", "q?")["answer"]       # the third succeeds and clears the failure
    assert len(calls) == 3 and d._failed is None


def test_an_input_that_is_neither_text_nor_json_is_a_reported_error():
    seen = []
    route = Router(ROUTES, "Which team?", model=decider(), fallback="human", on_error="fallback",
                   on_decision=seen.append)
    assert route(object()) == "human" and "text or JSON" in seen[0].error


def test_a_state_the_integration_cannot_read_takes_the_fallback_too():
    """Reading the framework's state (a LangGraph key, an Agno step input) is part of the decision: with
    on_error="fallback" a state that cannot be read takes the fallback, and the hooks see it."""
    pytest.importorskip("langchain_core")
    from opendecider.integrations.langchain import DecisionRouter
    seen = []
    route = DecisionRouter(ROUTES, "Which team?", model=decider(), state_key="ticket", fallback="human",
                           on_error="fallback", on_decision=seen.append)
    assert route({"something_else": "x"}) == "human"                 # no "ticket" key
    assert seen[0].reason == "error" and "KeyError" in seen[0].error
    strict = DecisionRouter(ROUTES, "Which team?", model=decider(), state_key="ticket", on_decision=seen.append)
    with pytest.raises(KeyError):
        strict({"something_else": "x"})
    assert seen[-1].reason == "error"                                # raised, and still reported


def test_an_exception_is_recorded_once_on_the_route_span(spans):
    route = Router(ROUTES, "Which team?", model=decider(), fallback="human")
    with pytest.raises(RuntimeError):
        route("boom")
    r = next(s for s in spans.get_finished_spans() if s.name == "opendecider.route")
    assert [e.name for e in r.events].count("exception") == 1


def test_an_outage_logs_once_per_window_not_once_per_request(caplog, monkeypatch):
    from opendecider import tools
    now = [500.0]
    monkeypatch.setattr(tools.time, "monotonic", lambda: now[0])
    route = Router(ROUTES, "Which team?", model=decider(), fallback="human", on_error="fallback")
    with caplog.at_level(logging.DEBUG, logger="opendecider.tools"):
        for _ in range(50):
            route("boom")
        now[0] += tools.LOG_REPEAT_S + 0.1
        route("boom")
    levels = [r.levelno for r in caplog.records if "routing failed" in r.getMessage()]
    assert levels.count(logging.ERROR) == 2 and levels.count(logging.DEBUG) == 49   # first, then after the window
