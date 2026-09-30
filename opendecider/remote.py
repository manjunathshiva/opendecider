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
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class RemoteModel:
    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: float = 120.0,
                 workers: int = 4):
        self.model, self.base_url = model, base_url.rstrip("/")
        url = urllib.parse.urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError(f"base_url must be an http or https URL, got {base_url!r}")
        self.api_key = api_key or os.environ.get("OPENDECIDER_REMOTE_API_KEY") or None
        if (self.api_key and url.scheme == "http" and url.hostname not in LOOPBACK
                and os.environ.get("OPENDECIDER_REMOTE_ALLOW_HTTP") != "1"):
            raise ValueError(f"refusing to send an API key over plain HTTP to {url.hostname}; use https, or set "
                             "OPENDECIDER_REMOTE_ALLOW_HTTP=1 if the network path is trusted")
        self.timeout, self.workers = timeout, max(1, workers)
        self.device = f"remote ({self.base_url})"

    def _complete(self, prompt: str) -> dict:
        body = {"model": self.model, "messages": [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": prompt}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
        req = self._request("/chat/completions", json.dumps(body).encode())
        for attempt in range(3):   # up to two retries, 0.5 s then 1 s apart
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                try:
                    return json.loads(raw)
                except ValueError:
                    raise RuntimeError(f"{self.base_url} answered with something that is not JSON: "
                                       f"{raw[:200].decode(errors='replace')!r}") from None
            except TimeoutError:
                raise RuntimeError(f"{self.base_url} gave no answer within {self.timeout:g} s") from None
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUS and attempt < 2:
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                raise RuntimeError(f"{self.base_url} answered HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
            except (urllib.error.URLError, ConnectionError) as e:
                if isinstance(getattr(e, "reason", e), ConnectionResetError) and attempt < 2:
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                raise RuntimeError(f"cannot reach {self.base_url} ({getattr(e, 'reason', e)}); "
                                   "is the server running and the model loaded?") from None
        raise RuntimeError(f"{self.base_url} did not answer after 3 attempts")   # not reached: each path returns or raises

    def _request(self, path: str, data: bytes | None = None) -> urllib.request.Request:
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"content-type": "application/json"} if data is not None else {})
        if self.api_key:   # an unredirected header: urllib never copies it onto a redirect to another origin
            req.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        return req

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


def parse(name: str, base_url: str | None = None) -> tuple[str, str] | None:
    """'lmstudio:model' / 'ollama:model' / 'openai:model' -> (model, base URL); None when it is not a remote name."""
    scheme, sep, model = name.partition(":")
    if not sep or scheme not in (*DEFAULT_URLS, "openai") or not model or model.startswith("//"):
        return None
    url = base_url or os.environ.get("OPENDECIDER_REMOTE_URL") or DEFAULT_URLS.get(scheme)
    if not url:
        raise ValueError("openai:<model> needs base_url=... (or OPENDECIDER_REMOTE_URL)")
    return model, url
