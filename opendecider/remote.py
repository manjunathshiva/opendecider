"""OpenDecider through a model server: LM Studio, Ollama, vLLM, or another server with an OpenAI-compatible
``/v1/chat/completions`` that returns ``top_logprobs``.

The server only runs the model (opendecider-small or -small-td: the GGUF build, or the base model with the LoRA
adapter). OpenDecider builds the same prompt the model was trained on and reads the probability of each option letter
from the next-token log-probabilities, so a Q8_0 GGUF build gives the same top answer as the PyTorch model on about 99%
of typed-decisions questions (1,975 of 2,000 for opendecider-small, at 0.669 vs 0.671 accuracy).

Tested with LM Studio (its llama.cpp engine) and Ollama (the GGUF builds), and with vLLM serving the base model with
the LoRA adapter (`--enable-lora --max-logprobs 20`): same accuracy as the PyTorch model on typed-decisions. Other
servers with an OpenAI-compatible chat endpoint that returns `top_logprobs` should work the same way. LM Studio's MLX
engine returns no log-probabilities.

    from opendecider import load
    model = load("lmstudio:opendecider-small")                 # LM Studio on http://127.0.0.1:1234
    model = load("ollama:opendecider-small")                   # Ollama on http://127.0.0.1:11434
    model = load("openai:opendecider-small", base_url="http://gpu-box:8000/v1")   # e.g. vLLM

    opendecider serve --model lmstudio:opendecider-small       # Jev-compatible /v1/systemone on top of LM Studio

Server URL: `base_url=...`, else OPENDECIDER_REMOTE_URL (for any of the three prefixes), else LM Studio's or Ollama's
default local address.

`opendecider serve` itself: pass its URL as the model (`load("https://decider.internal:8000")`). Every question goes to
the server as the same named options the local model would see, so the answers are the server model's own, and the
client needs no model, GPU or download.

Credentials: set OPENDECIDER_REMOTE_API_KEY if the server needs one. It is sent only to that origin (never on a
redirect), and over plain HTTP only to this machine; for another host use https, or set
OPENDECIDER_REMOTE_ALLOW_HTTP=1 if the network path is trusted.

Limits: up to 26 options per question (letters A to Z). OpenAI-compatible servers return at most 20 top
log-probabilities, so with 21 to 26 options, a letter outside the top 20 gets a probability close to zero.
"""
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .prompt import LETTERS, SYSTEM, render

DEFAULT_URLS = {"lmstudio": "http://127.0.0.1:1234/v1", "ollama": "http://127.0.0.1:11434/v1"}
RETRY_STATUS = {429, 500, 502, 503, 504}   # a busy or restarting server: try again before failing the request
DOWN_RETRY_S = 5.0   # after a server times out or cannot be reached, calls fail at once for this long
_clock = time.monotonic   # the time source for the fail-fast window (tests replace it, not time.monotonic)
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class ServerError(RuntimeError):
    """The server failed to answer (unreachable, timed out, rejected the request, or answered something unusable)."""


class _Client:
    """HTTP to one server: the URL and credential rules, and retries for a busy or restarting server."""

    thread_safe = True        # each request is independent: callers need not take turns
    retry_status = RETRY_STATUS

    def __init__(self, base_url: str, api_key: str | None, timeout: float):
        self.base_url = base_url.rstrip("/")
        url = urllib.parse.urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError(f"base_url must be an http or https URL, got {base_url!r}")
        self.api_key = api_key or os.environ.get("OPENDECIDER_REMOTE_API_KEY") or None
        if (self.api_key and url.scheme == "http" and url.hostname not in LOOPBACK
                and os.environ.get("OPENDECIDER_REMOTE_ALLOW_HTTP") != "1"):
            raise ValueError(f"refusing to send an API key over plain HTTP to {url.hostname}; use https, or set "
                             "OPENDECIDER_REMOTE_ALLOW_HTTP=1 if the network path is trusted")
        self.timeout = timeout
        self._down: tuple[float, str] | None = None   # (retry after, message) once the server timed out or vanished

    def _post(self, path: str, body: dict) -> dict:
        down = self._down
        if down and _clock() < down[0]:   # a hung server costs one timeout, not one per request
            raise ServerError(down[1])
        try:
            out = self._send(path, body)
        except ServerError as e:
            if getattr(e, "unreachable", False):
                self._down = (_clock() + DOWN_RETRY_S, f"{e} (calls fail at once for {DOWN_RETRY_S:g} s)")
            raise
        self._down = None
        return out

    def _send(self, path: str, body: dict) -> dict:
        req = self._request(path, json.dumps(body).encode())
        for attempt in range(3):   # up to two retries, 0.5 s then 1 s apart (or the server's Retry-After, up to 5 s)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                try:
                    return json.loads(raw)
                except ValueError:
                    raise ServerError(f"{self.base_url} answered with something that is not JSON: "
                                      f"{raw[:200].decode(errors='replace')!r}") from None
            except TimeoutError:
                raise _unreachable(f"{self.base_url} gave no answer within {self.timeout:g} s") from None
            except urllib.error.HTTPError as e:
                if e.code in self.retry_status and attempt < 2:
                    time.sleep(_retry_after(e, 0.5 * 2 ** attempt))
                    continue
                self._http_error(e)
            except (urllib.error.URLError, ConnectionError) as e:
                if isinstance(getattr(e, "reason", e), ConnectionResetError) and attempt < 2:
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                if isinstance(getattr(e, "reason", None), TimeoutError):   # urllib wraps a connect timeout
                    raise _unreachable(f"{self.base_url} gave no answer within {self.timeout:g} s") from None
                raise _unreachable(f"cannot reach {self.base_url} ({getattr(e, 'reason', e)}); "
                                   "is the server running and the model loaded?") from None
        raise ServerError(f"{self.base_url} did not answer after 3 attempts")   # not reached: each path returns or raises

    def _http_error(self, e: "urllib.error.HTTPError"):
        raise ServerError(f"{self.base_url} answered HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None

    def _request(self, path: str, data: bytes | None = None) -> urllib.request.Request:
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"content-type": "application/json"} if data is not None else {})
        if self.api_key:   # an unredirected header: urllib never copies it onto a redirect to another origin
            req.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        return req


def _unreachable(message: str) -> ServerError:
    """A ServerError for a server that timed out or could not be reached (it trips the fail-fast window)."""
    e = ServerError(message)
    e.unreachable = True
    return e


def _retry_after(e: "urllib.error.HTTPError", default: float) -> float:
    """The server's Retry-After in seconds (capped at 5), else `default`."""
    try:
        return min(5.0, max(0.0, float((getattr(e, "headers", None) or {}).get("Retry-After"))))
    except (TypeError, ValueError):
        return default


class RemoteModel(_Client):
    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: float = 120.0,
                 workers: int = 4):
        super().__init__(base_url, api_key, timeout)
        self.model = model
        self.workers = max(1, workers)
        self.device = f"remote ({self.base_url})"

    def _complete(self, prompt: str) -> dict:
        body = {"model": self.model, "messages": [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": prompt}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
        return self._post("/chat/completions", body)

    def ping(self) -> bool:
        """True when the server answers GET /models (used by `opendecider serve`'s /ready)."""
        req = self._request("/models")
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return False

    def _decide(self, state, instructions: str, options: dict) -> tuple[dict, int]:
        names = list(options)
        if len(names) > len(LETTERS):
            raise ValueError(f"at most {len(LETTERS)} options per question through a model server (got {len(names)}); "
                             "use the PyTorch model for more")
        r = self._complete(render(state, instructions, options))
        try:
            top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"{self.base_url} returned no token log-probabilities for {self.model!r}. Use a server "
                               "and engine that supports them (LM Studio and Ollama with the GGUF build, and vLLM, do; "
                               "LM Studio's MLX engine does not)") from None
        # "A" and " A" are different tokens that both mean the letter A: add their probabilities
        prob: dict = {}
        for t in top:
            letter = t["token"].strip()
            prob[letter] = prob.get(letter, 0.0) + math.exp(t["logprob"])
        wanted = LETTERS[:len(names)]
        if not any(c in prob for c in wanted):
            raise RuntimeError(f"none of the option letters {wanted[0]}-{wanted[-1]} is among the most likely next tokens "
                               f"from {self.model!r}; is this an OpenDecider model (opendecider-small or -small-td)?")
        # a letter outside the top 20: well below the least likely token returned
        floor = min(math.exp(t["logprob"]) for t in top) * math.exp(-5.0)
        w = [prob.get(c, floor) for c in wanted]
        z = sum(w)
        tokens = (r.get("usage") or {}).get("prompt_tokens", 0)
        return {k: v / z for k, v in zip(names, w)}, tokens

    def decide_many(self, items: list[tuple], info: list | None = None) -> list[dict]:
        with ThreadPoolExecutor(max_workers=min(self.workers, len(items) or 1)) as pool:
            res = list(pool.map(lambda it: self._decide(*it), items))
        if info is not None:
            info.extend({"input_tokens": t, "truncated": False} for _, t in res)
        return [p for p, _ in res]


class ServedModel(_Client):
    """A model behind `opendecider serve`, by URL: each question goes to the server as the named options the local
    model would see, so answers match the server model's own exactly. Questions about one state go in one request
    (up to `max_questions`, the server's default limit); several states go out in parallel (`workers`)."""

    retry_status = {429, 502, 503}   # not 504: the server already waited its own request timeout

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0, workers: int = 4,
                 max_questions: int = 64):
        super().__init__(base_url, api_key, timeout)
        self.workers, self.max_questions = max(1, workers), max(1, max_questions)
        self.device = f"served ({self.base_url})"
        try:   # fail now, with a clear message, rather than on the first decision
            with urllib.request.urlopen(self._request("/v1/models"), timeout=min(self.timeout, 10)) as r:
                listed = (json.loads(r.read()).get("models") or [{}])[0]
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError, AttributeError,
                IndexError) as e:
            raise ServerError(f"cannot reach opendecider serve at {self.base_url} "
                              f"({getattr(e, 'reason', e)}); is it running?") from None
        if not isinstance(listed, dict):   # an unexpected listing: use the URL as the name
            listed = {}
        self.name = listed.get("name") or self.base_url
        self.kind = listed.get("kind")

    def _http_error(self, e: "urllib.error.HTTPError"):
        detail = e.read()[:300].decode(errors="replace")
        try:
            body = json.loads(detail)
        except ValueError:   # not JSON (a proxy's HTML page, say): report the text as it came
            body = None
        if isinstance(body, dict) and body.get("detail"):   # the server's own message, e.g. FastAPI's {"detail": ...}
            detail = body["detail"]
        if e.code in (400, 413, 422):   # the request itself: a caller's input error, as with a local model
            raise ValueError(f"{self.base_url} rejected the request: {detail}") from None
        if e.code in (401, 403):
            raise ServerError(f"{self.base_url} rejected the credentials (HTTP {e.code}); set "
                              "OPENDECIDER_REMOTE_API_KEY or pass api_key=") from None
        raise ServerError(f"{self.base_url} answered HTTP {e.code}: {detail}") from None

    def ping(self) -> bool:
        """True when the server reports ready (GET /ready)."""
        try:
            with urllib.request.urlopen(self._request("/ready"), timeout=3) as r:
                return r.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return False

    def _ask(self, state, group: list[tuple]) -> tuple[list[dict], list[dict]]:
        questions = {f"q{j}": {"type": "choice", "instructions": instr, "criteria": opts}
                     for j, (_, instr, opts) in enumerate(group)}
        r = self._post("/v1/systemone", {"state": state, "questions": questions})
        probs, info = [], []
        for j, (_, _, opts) in enumerate(group):
            try:
                a = r["answers"][f"q{j}"]
                probs.append({name: float(a["probabilities"][name]) for name in opts})
            except (KeyError, TypeError, ValueError):
                raise ServerError(f"{self.base_url} returned no probabilities for every option; is it opendecider "
                                  "serve (a Jev or Laya server returns only the top answer)?") from None
            info.append({"input_tokens": 0, "truncated": bool(a.get("truncated"))})
        info[0]["input_tokens"] = int((r.get("usage") or {}).get("input_tokens") or 0)   # the request's total
        return probs, info

    def decide_many(self, items: list[tuple], info: list | None = None) -> list[dict]:
        groups: list[tuple] = []   # consecutive questions about the same state share a request
        for it in items:
            if groups and groups[-1][0] is it[0] and len(groups[-1][1]) < self.max_questions:
                groups[-1][1].append(it)
            else:
                groups.append((it[0], [it]))
        if len(groups) == 1:   # one request (a routing call): no thread pool
            res = [self._ask(*groups[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(groups))) as pool:
                res = list(pool.map(lambda g: self._ask(*g), groups))
        if info is not None:
            info.extend(i for _, inf in res for i in inf)
        return [p for ps, _ in res for p in ps]


def is_url(name: str) -> bool:
    """True for an `opendecider serve` URL given as the model name."""
    return isinstance(name, str) and name.lower().startswith(("http://", "https://"))


def parse(name: str, base_url: str | None = None) -> tuple[str, str] | None:
    """'lmstudio:model' / 'ollama:model' / 'openai:model' -> (model, base URL); None when it is not a remote name."""
    scheme, sep, model = name.partition(":")
    if not sep or scheme not in (*DEFAULT_URLS, "openai") or not model or model.startswith("//"):
        return None
    url = base_url or os.environ.get("OPENDECIDER_REMOTE_URL") or DEFAULT_URLS.get(scheme)
    if not url:
        raise ValueError("openai:<model> needs base_url=... (or OPENDECIDER_REMOTE_URL)")
    return model, url
