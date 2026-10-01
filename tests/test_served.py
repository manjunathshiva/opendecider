"""A model behind `opendecider serve`, used by URL: a real server on a local port with a fake model (no download)."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from opendecider import OpenDecider, load  # noqa: E402
from opendecider.remote import ServedModel, ServerError  # noqa: E402
from opendecider.serve import Settings, create_app  # noqa: E402


class Fake:
    """The option named in the state gets 0.6; the rest share 0.4 unevenly, so every probability differs."""

    def __init__(self):
        self.device = "cpu"

    def decide_many(self, items, info=None):
        out = []
        for state, _, opts in items:
            names = list(opts)
            top = next((n for n in names if n.lower() in str(state).lower()), names[0])
            rest = [n for n in names if n != top]
            weights = [2 ** -i for i in range(1, len(rest) + 1)]
            out.append({top: 0.6, **{n: 0.4 * w / sum(weights) for n, w in zip(rest, weights)}})
            if info is not None:
                info.append({"input_tokens": 7, "truncated": "very long" in str(state)})
        return out


def local():
    return OpenDecider(Fake(), {"name": "opendecider-test", "kind": "nano"})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def serve():
    """Start `opendecider serve` (fake model) on a free port; yields a function taking Settings overrides."""
    servers = []

    def start(**kw):
        port = _free_port()
        app = create_app(model=local(), settings=Settings(**kw))
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=srv.run, daemon=True).start()
        for _ in range(200):
            if srv.started:
                break
            time.sleep(0.02)
        servers.append(srv)
        return f"http://127.0.0.1:{port}"

    yield start
    for srv in servers:
        srv.should_exit = True


QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs",
                                                                        "sales": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "churn": {"type": "noul", "instructions": "Will they leave?"},
}


def test_answers_match_the_local_model_exactly(serve):
    url = serve()
    served = load(url)
    assert served.name == "opendecider-test" and served.meta["kind"] == "served"
    for state in ["a billing problem", {"ticket": "tech outage", "tier": "gold"}, "very long text"]:
        r, want = served.system_one(state, QUESTIONS), local().system_one(state, QUESTIONS)
        assert r["answers"] == want["answers"]   # every probability, choice, score, expected and noul
    many = served.system_one_batch(["billing", "tech", "sales"], QUESTIONS)
    assert [m["answers"] for m in many] == [a["answers"] for a in local().system_one_batch(["billing", "tech",
                                                                                            "sales"], QUESTIONS)]
    assert served.system_one("very long text", QUESTIONS)["answers"]["team"]["truncated"] is True


def test_routers_and_tools_take_a_url(serve):
    from opendecider import tools
    url = serve()
    route = tools.Router({"billing": "charges", "tech": "bugs"}, "Which team?", model=url, fallback="human",
                         min_confidence=0.5)
    assert route("the tech dashboard is down") == "tech"
    assert tools.choose(tools.shared(url), "billing", "Which?", ["billing", "tech"])["choice"] == "billing"


def test_more_questions_than_the_server_takes_per_request_are_split(serve):
    url = serve(max_questions=2)
    qs = {f"q{i}": {"type": "noul", "instructions": f"question {i}?"} for i in range(5)}
    m = OpenDecider(ServedModel(url, max_questions=2), {"name": "x", "kind": "served"})
    assert len(m.system_one("s", qs)["answers"]) == 5
    with pytest.raises(ValueError, match="at most 2 questions"):   # the server's own 422, as a caller's input error
        OpenDecider(ServedModel(url), {"name": "x", "kind": "served"}).system_one("s", qs)


def test_credentials(serve, monkeypatch):
    monkeypatch.delenv("OPENDECIDER_REMOTE_API_KEY", raising=False)
    url = serve(api_key="sekret")
    with pytest.raises(ServerError, match="rejected the credentials"):
        load(url).system_one("s", QUESTIONS)
    assert load(url, api_key="sekret").system_one("billing", QUESTIONS)["answers"]["team"]["choice"] == "billing"
    monkeypatch.setenv("OPENDECIDER_REMOTE_API_KEY", "sekret")
    assert load(url).system_one("tech", QUESTIONS)["answers"]["team"]["choice"] == "tech"
    with pytest.raises(ValueError, match="plain HTTP"):   # a key never crosses the network in clear
        ServedModel("http://decider.lan:8000", api_key="sekret")


def test_unreachable_server_is_a_clear_error_and_a_model_error_for_agents():
    from opendecider import tools
    with pytest.raises(ServerError, match="cannot reach opendecider serve"):
        load(f"http://127.0.0.1:{_free_port()}")
    d = tools.Decider(f"http://127.0.0.1:{_free_port()}")
    assert "cannot reach" in tools.as_result(tools.yes_no, d, "s", "q?")["error"]


def test_a_server_without_probabilities_is_refused():
    """A Jev or Laya server speaks the same protocol but returns only the top answer."""
    from http.server import HTTPServer

    class Jev(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send({"models": [{"name": "jev"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self._send({"answers": {k: {"type": "choice", "choice": "x"} for k in body["questions"]}})

    srv = HTTPServer(("127.0.0.1", 0), Jev)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ServerError, match="no probabilities"):
            load(f"http://127.0.0.1:{srv.server_address[1]}").system_one("s", QUESTIONS)
    finally:
        srv.shutdown()


def test_busy_server_is_retried_after_its_retry_after(serve, monkeypatch):
    from opendecider import remote
    url = serve()
    m = ServedModel(url)
    calls, real = [], remote.urllib.request.urlopen

    def busy_once(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise remote.urllib.error.HTTPError(req.full_url, 503, "busy", {"Retry-After": "0"}, None)
        return real(req, timeout=timeout)

    monkeypatch.setattr(remote.urllib.request, "urlopen", busy_once)
    assert OpenDecider(m, {"name": "x", "kind": "served"}).system_one("billing", QUESTIONS)["answers"]
    assert len(calls) == 2


def test_a_server_timeout_is_not_retried(serve, monkeypatch):
    """The server answers 504 after its own request timeout: retrying would multiply the wait."""
    from opendecider import remote
    m = ServedModel(serve())
    calls = []

    def timed_out(req, timeout):
        calls.append(req.full_url)
        raise remote.urllib.error.HTTPError(req.full_url, 504, "timeout", {}, None)

    monkeypatch.setattr(remote.urllib.request, "urlopen", timed_out)
    with pytest.raises(ServerError, match="HTTP 504"):
        OpenDecider(m, {"name": "x", "kind": "served"}).system_one("s", QUESTIONS)
    assert len(calls) == 1


def test_server_error_bodies_become_readable_messages(monkeypatch):
    """FastAPI's {"detail": ...}, a proxy's HTML page and a JSON list all give a readable message."""
    import io
    from opendecider import remote
    m = ServedModel.__new__(ServedModel)
    m.base_url = "http://decider:8000"

    def err(code, body):
        return remote.urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body))

    with pytest.raises(ValueError, match="rejected the request: at most 2 questions"):
        m._http_error(err(422, b'{"detail": "at most 2 questions"}'))
    with pytest.raises(ServerError, match="HTTP 502: <html>bad gateway</html>"):
        m._http_error(err(502, b"<html>bad gateway</html>"))
    with pytest.raises(ServerError, match=r"HTTP 500: \[1, 2\]"):
        m._http_error(err(500, b"[1, 2]"))
