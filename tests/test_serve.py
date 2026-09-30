"""HTTP server: Jev wire compatibility, limits, auth, batching and back-pressure, with a fake model (no download)."""
import asyncio
import threading
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from opendecider import OpenDecider  # noqa: E402
from opendecider.serve import Batcher, Settings, create_app  # noqa: E402


class Fake:
    """Deterministic: the first option gets 0.7, the rest share 0.3. Records each batch size."""

    def __init__(self, delay=0.0, gate=None):
        self.batches, self.delay, self.gate, self.device = [], delay, gate, "cpu"

    def decide_many(self, items, info=None):
        if self.gate is not None:
            self.gate.wait()
        time.sleep(self.delay)
        self.batches.append(len(items))
        out = []
        for state, _, opts in items:
            if state == "boom":
                raise RuntimeError("model failure")
            if state == "too long":
                raise ValueError("input is 99999 tokens, longer than this model's 32768-token context")
            names = list(opts)
            rest = 0.3 / (len(names) - 1)
            # "malformed": a model bug (no probabilities), so building the answer fails after inference
            out.append({} if state == "malformed" else {k: 0.7 if i == 0 else rest for i, k in enumerate(names)})
            if info is not None:
                info.append({"input_tokens": 10, "truncated": state == "very long"})
        return out


def make(fake=None, **kw):
    fake = fake or Fake()
    model = OpenDecider(fake, {"name": "opendecider-test", "kind": "nano"})
    return create_app(model=model, settings=Settings(**kw)), fake


def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


QS = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs"}},
      "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
      "churn": {"type": "noul", "instructions": "Will they cancel?"}}


def test_jev_wire_shape():
    app, _ = make()
    r = client(app).post("/v1/systemone", json={"model": "typesafe/jev-latest", "state": "billed twice", "questions": QS})
    assert r.status_code == 200
    b = r.json()
    assert set(b) >= {"model", "answers", "usage"} and b["model"] == "opendecider-test"
    assert b["usage"] == {"input_tokens": 30, "output_tokens": 0}
    assert b["answers"]["team"]["type"] == "choice" and b["answers"]["team"]["choice"] == "billing"
    u = b["answers"]["urgency"]   # Jev semantics: score = expected score, level = most likely level
    assert u["score"] == pytest.approx(0.3) and u["level"] == 0 and u["legend"] == {"0": "low", "1": "high"}
    assert "expected" not in u and r.headers["x-request-id"]
    assert b["answers"]["churn"]["type"] == "noul" and b["answers"]["churn"]["noul"] == pytest.approx(0.7)
    assert "warnings" not in b and "Server-Timing" in r.headers


def test_truncation_is_reported():
    app, _ = make()
    b = client(app).post("/v1/systemone", json={"state": "very long", "questions": QS}).json()
    assert b["answers"]["team"]["truncated"] is True and len(b["warnings"]) == 3


def test_batch_endpoint():
    app, fake = make()
    r = client(app).post("/v1/systemone/batch", json={"states": ["a", {"x": 1}, "c"], "questions": QS})
    assert r.status_code == 200
    b = r.json()
    assert len(b["results"]) == 3 and b["total_usage"]["input_tokens"] == 90
    assert fake.batches == [9]   # all states and questions in one model call


def test_validation_errors():
    app, _ = make(max_questions=2, max_options=3, max_state_chars=50, max_body_bytes=2000)
    c = client(app)
    assert c.post("/v1/systemone", content=b"not json").status_code == 400
    assert c.post("/v1/systemone", json=[1, 2]).status_code == 400
    assert c.post("/v1/systemone", json={"state": "x"}).status_code == 422
    assert c.post("/v1/systemone", json={"state": None, "questions": {"q": QS["team"]}}).status_code == 200  # null state, as Jev
    bad = {"q": {"type": "choice", "instructions": "?", "criteria": ["only one"]}}
    r = c.post("/v1/systemone", json={"state": "x", "questions": bad})
    assert r.status_code == 422 and "'q'" in r.json()["detail"]
    assert c.post("/v1/systemone", json={"state": "x", "questions": QS}).status_code == 422          # 3 > 2 questions
    many = {"q": {"type": "choice", "instructions": "?", "criteria": list("abcd")}}
    assert c.post("/v1/systemone", json={"state": "x", "questions": many}).status_code == 422        # 4 > 3 options
    one = {"team": QS["team"]}
    assert c.post("/v1/systemone", json={"state": "x" * 60, "questions": one}).status_code == 422    # state too long
    assert c.post("/v1/systemone", json={"state": "x" * 3000, "questions": one}).status_code == 413  # body too large


def test_auth():
    app, _ = make(api_key="s3cret")
    c = client(app)
    body = {"state": "x", "questions": {"team": QS["team"]}}
    assert c.post("/v1/systemone", json=body).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer s\xe9cret".encode("latin-1")}).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/health").status_code == 200   # probes stay open


def test_model_failure_is_isolated_and_logged():
    app, _ = make()
    c = client(app)
    r = c.post("/v1/systemone", json={"state": "boom", "questions": {"team": QS["team"]}})
    assert r.status_code == 500 and r.json()["detail"] == "inference failed"   # no internals leaked
    assert c.post("/v1/systemone", json={"state": "fine", "questions": {"team": QS["team"]}}).status_code == 200
    r = c.post("/v1/systemone", json={"state": "too long", "questions": {"team": QS["team"]}})
    assert r.status_code == 422 and "context" in r.json()["detail"]   # the client's input problem, named



def test_unexpected_error_keeps_json_body_and_request_id(caplog):
    app, _ = make()
    c = client(app)
    r = c.post("/v1/systemone", json={"state": "malformed", "questions": {"churn": QS["churn"]}},
               headers={"x-request-id": "req-42"})
    assert r.status_code == 500 and r.json() == {"detail": "internal server error"}   # no internals leaked
    assert r.headers["x-request-id"] == "req-42"
    assert "req-42" in caplog.text and "Traceback" in caplog.text   # the details go to the server log, with the id
    assert c.post("/v1/systemone", json={"state": "fine", "questions": {"churn": QS["churn"]}}).status_code == 200
    assert 'opendecider_requests_total{endpoint="systemone",code="500"} 1' in c.get("/metrics").text


def test_health_ready_metrics():
    app, _ = make()
    c = client(app)
    c.post("/v1/systemone", json={"state": "x", "questions": QS})
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "opendecider-test" and h["kind"] == "nano"
    assert c.get("/ready").json() == {"ready": True}
    assert c.get("/v1/models").json()["models"][0]["name"] == "opendecider-test"
    m = c.get("/metrics").text
    assert 'opendecider_requests_total{endpoint="systemone",code="200"} 1' in m
    assert "opendecider_questions_total 3" in m and "opendecider_request_seconds_bucket" in m


def test_batcher_merges_concurrent_requests():
    fake = Fake(delay=0.05)
    model = OpenDecider(fake, {"name": "t", "kind": "nano"})
    b = Batcher(model, max_batch=64, wait_ms=20)
    items = model.items("x", model.prepare(QS))
    futs = [b.submit(items) for _ in range(8)]
    res = [f.result(timeout=5) for f in futs]
    assert all(len(p) == 3 for p, _ in res)
    assert sum(fake.batches) == 24 and len(fake.batches) < 8   # merged into fewer model calls


def test_batcher_splits_large_requests():
    fake = Fake()
    model = OpenDecider(fake, {"name": "t", "kind": "nano"})
    b = Batcher(model, max_batch=4, wait_ms=0)
    items = model.items("x", model.prepare(QS)) * 3   # 9 questions
    probs, info = b.submit(items).result(timeout=5)
    assert len(probs) == 9 and fake.batches == [4, 4, 1]


def test_backpressure_and_timeout():
    gate = threading.Event()
    app, fake = make(Fake(gate=gate), max_in_flight=2, request_timeout_s=0.5)
    transport = httpx.ASGITransport(app=app)
    body = {"state": "x", "questions": {"team": QS["team"]}}

    async def go():
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            tasks = [asyncio.create_task(c.post("/v1/systemone", json=body)) for _ in range(2)]
            await asyncio.sleep(0.1)
            busy = await c.post("/v1/systemone", json=body)          # third while two are admitted
            done = await asyncio.gather(*tasks)
            return busy, done

    busy, done = asyncio.run(go())
    gate.set()
    assert busy.status_code == 503 and busy.headers["Retry-After"] == "1"
    assert [r.status_code for r in done] == [504, 504]                # model blocked past the timeout


def test_ready_reflects_an_unreachable_upstream():
    fake = Fake()
    fake.ping = lambda: False          # e.g. LM Studio / Ollama not running behind `opendecider serve`
    app, _ = make(fake)
    r = client(app).get("/ready")
    assert r.status_code == 503 and r.json()["upstream"] is False
