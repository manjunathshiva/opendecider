"""The LM Studio / Ollama backend against a fake OpenAI-compatible server (no model, no network beyond localhost)."""
import json
import math
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from opendecider import load
from opendecider.remote import parse


class Handler(BaseHTTPRequestHandler):
    mode = "ok"
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Handler.seen.append(body)
        if Handler.mode == "nologprobs":
            out = {"choices": [{"index": 0, "message": {"content": ""}, "logprobs": None}]}
        else:
            top = [{"token": "A", "logprob": -0.1}, {"token": " B", "logprob": -2.5}, {"token": "Z", "logprob": -3.0}]
            out = {"choices": [{"index": 0, "message": {"content": "A"},
                                "logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": top}]}}],
                   "usage": {"prompt_tokens": 42, "completion_tokens": 1}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Server(HTTPServer):
    def server_bind(self):   # skip HTTPServer's reverse-DNS lookup, which can take half a minute on macOS
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


@pytest.fixture()
def server():
    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Handler.mode, Handler.seen = "ok", []
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_parse_names():
    assert parse("lmstudio:opendecider-small") == ("opendecider-small", "http://127.0.0.1:1234/v1")
    assert parse("ollama:opendecider-small-td") == ("opendecider-small-td", "http://127.0.0.1:11434/v1")
    assert parse("openai:m", "http://x:8000/v1") == ("m", "http://x:8000/v1")
    assert parse("manjunathshiva/opendecider-nano") is None and parse("C:/models/nano") is None
    with pytest.raises(ValueError):
        parse("openai:m")


def test_answers_from_logprobs(server):
    m = load("openai:opendecider-small", base_url=server)
    r = m.system_one("billed twice", {"team": {"type": "choice", "instructions": "Which team?",
                                               "criteria": {"billing": "charges", "tech": "bugs", "other": None}},
                                      "urgent": {"type": "noul", "instructions": "Urgent?"}})
    team = r["answers"]["team"]["probabilities"]
    assert r["answers"]["team"]["choice"] == "billing" and team["billing"] > team["tech"] > team["other"] > 0
    assert r["answers"]["urgent"]["noul"] > 0.5 and r["usage"]["input_tokens"] == 84
    assert len(Handler.seen) == 2   # one request per question (sent in parallel, so in any order)
    sent = next(b for b in Handler.seen if "Which team?" in b["messages"][1]["content"])
    assert sent["model"] == "opendecider-small" and sent["max_tokens"] == 1 and sent["top_logprobs"] == 20
    assert sent["messages"][0]["content"] == "You make one decision for a software system."
    assert "A) billing: charges" in sent["messages"][1]["content"]   # the trained prompt, not the server's own


def test_clear_errors(server):
    m = load("openai:x", base_url=server)
    Handler.mode = "nologprobs"
    with pytest.raises(RuntimeError, match="log-probabilities"):
        m.system_one("s", {"q": {"type": "noul", "instructions": "?"}})
    with pytest.raises(ValueError, match="26 options"):
        m.system_one("s", {"q": {"type": "choice", "instructions": "?", "criteria": [f"o{i}" for i in range(30)]}})
    with pytest.raises(RuntimeError, match="cannot reach"):
        load("openai:x", base_url="http://127.0.0.1:9/v1").system_one("s", {"q": {"type": "noul", "instructions": "?"}})


def _server_with(top, status=200):
    """A one-shot fake server returning these top_logprobs (or an HTTP status)."""
    class H(BaseHTTPRequestHandler):
        calls = 0

        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            H.calls += 1
            if H.calls <= status_fail[0]:
                self.send_response(503); self.send_header("content-length", "4"); self.end_headers(); self.wfile.write(b"busy")
                return
            out = {"choices": [{"index": 0, "message": {"content": "A"},
                                "logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": top}]}}],
                   "usage": {"prompt_tokens": 7}}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            data = b'{"data":[]}'
            self.send_response(200); self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
    status_fail = [0 if status == 200 else status]
    srv = Server(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1", H


def test_same_letter_tokens_are_summed_not_overwritten():
    # "A" and " A" are different tokens; the answer "A" has the probability of both
    top = [{"token": "B", "logprob": math.log(0.5)}, {"token": "A", "logprob": math.log(0.3)},
           {"token": " A", "logprob": math.log(0.2)}]
    srv, url, _ = _server_with(top)
    try:
        p = load("openai:x", base_url=url).system_one("s", {"q": {"type": "choice", "instructions": "?",
                                                                   "criteria": ["a", "b"]}})["answers"]["q"]["probabilities"]
        assert p["a"] == pytest.approx(0.5) and p["b"] == pytest.approx(0.5)
    finally:
        srv.shutdown()


def test_no_option_letters_is_an_error_not_a_uniform_answer():
    srv, url, _ = _server_with([{"token": "Hello", "logprob": -0.1}, {"token": "The", "logprob": -1.0}])
    try:
        with pytest.raises(RuntimeError, match="none of the option letters"):
            load("openai:x", base_url=url).system_one("s", {"q": {"type": "noul", "instructions": "?"}})
    finally:
        srv.shutdown()


def test_transient_upstream_errors_are_retried():
    srv, url, H = _server_with([{"token": "A", "logprob": -0.1}], status=2)   # 503 twice, then answers
    try:
        r = load("openai:x", base_url=url).system_one("s", {"q": {"type": "noul", "instructions": "?"}})
        assert r["answers"]["q"]["noul"] > 0.5 and H.calls == 3
    finally:
        srv.shutdown()


def test_ping():
    srv, url, _ = _server_with([{"token": "A", "logprob": -0.1}])
    try:
        assert load("openai:x", base_url=url).impl.ping() is True
    finally:
        srv.shutdown()
    assert load("openai:x", base_url="http://127.0.0.1:9/v1").impl.ping() is False
