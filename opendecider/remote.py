"""OpenDecider through a local model server: LM Studio, Ollama, llama.cpp's server, vLLM, or anything with an
OpenAI-compatible ``/v1/chat/completions`` that returns ``top_logprobs``.

The server only runs the model (the GGUF build of opendecider-small or -small-td). OpenDecider builds the same prompt
the model was trained on and reads the probability of each option letter from the next-token log-probabilities, so
the answers match the PyTorch model (typed-decisions: 0.669 through a Q8_0 GGUF vs 0.671 in PyTorch).

    from opendecider import load
    model = load("lmstudio:opendecider-small")                 # LM Studio on http://127.0.0.1:1234
    model = load("ollama:opendecider-small")                   # Ollama on http://127.0.0.1:11434
    model = load("openai:opendecider-small", base_url="http://gpu-box:8000/v1")

    opendecider serve --model lmstudio:opendecider-small       # Jev-compatible /v1/systemone on top of LM Studio

Limits: up to 26 options per question (letters A to Z). OpenAI-compatible servers return at most 20 top
log-probabilities, so with 21 to 26 options, a letter outside the top 20 gets a probability close to zero.
"""
import json
import math
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .prompt import LETTERS, SYSTEM, render

DEFAULT_URLS = {"lmstudio": "http://127.0.0.1:1234/v1", "ollama": "http://127.0.0.1:11434/v1"}


class RemoteModel:
    def __init__(self, model: str, base_url: str, api_key: str | None = None, timeout: float = 120.0,
                 workers: int = 4):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENDECIDER_REMOTE_API_KEY") or "not-needed"
        self.timeout, self.workers = timeout, max(1, workers)
        self.device = f"remote ({self.base_url})"

    def _complete(self, prompt: str) -> dict:
        body = {"model": self.model, "messages": [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": prompt}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json",
                                              "authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{self.base_url} answered HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"cannot reach {self.base_url} ({e.reason}); is the server running and the model loaded?") from None

    def decide(self, state, instructions: str, options: dict) -> tuple[dict, int]:
        names = list(options)
        if len(names) > len(LETTERS):
            raise ValueError(f"at most {len(LETTERS)} options per question through a model server (got {len(names)}); "
                             "use the PyTorch model for more")
        r = self._complete(render(state, instructions, options))
        try:
            top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"{self.base_url} returned no token log-probabilities for {self.model!r}. Use a server "
                               "and engine that supports them (LM Studio and Ollama with the GGUF build do; LM Studio's "
                               "MLX engine does not)") from None
        lp = {t["token"].strip(): t["logprob"] for t in top}
        floor = min(lp.values(), default=0.0) - 5.0   # a letter outside the top 20: far below everything returned
        scores = [lp.get(LETTERS[i], floor) for i in range(len(names))]
        m = max(scores)
        w = [math.exp(s - m) for s in scores]
        z = sum(w)
        tokens = (r.get("usage") or {}).get("prompt_tokens", 0)
        return {k: v / z for k, v in zip(names, w)}, tokens

    def decide_many(self, items: list[tuple], info: list | None = None) -> list[dict]:
        with ThreadPoolExecutor(max_workers=min(self.workers, len(items) or 1)) as pool:
            res = list(pool.map(lambda it: self.decide(*it), items))
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
