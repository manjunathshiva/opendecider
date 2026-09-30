"""HTTP server for OpenDecider, speaking TypeSafe Jev's ``/v1/systemone`` wire protocol.

A client written for Jev (or for Laya's server) keeps working when pointed at this server:
the request is ``{"model"?, "state", "questions"}`` and the response is
``{"model", "answers", "usage"}`` with the same typed answers (``choice`` / ``score`` /
``noul``). OpenDecider adds fields a Jev client ignores (``probabilities`` and ``confidence``
on every answer, ``warnings`` when an input was truncated).

    pip install "opendecider[serve]"
    opendecider serve --model manjunathshiva/opendecider-nano          # http://0.0.0.0:8000

Endpoints: ``POST /v1/systemone``, ``POST /v1/systemone/batch``, ``GET /health``,
``GET /ready``, ``GET /metrics`` (Prometheus text format).

Concurrency: requests are admitted up to ``max_in_flight``; beyond that the server answers
503 with ``Retry-After`` at once instead of queueing without bound. Admitted requests are
merged by a dynamic batcher: one inference thread takes the questions of every waiting
request (up to ``max_batch`` questions, waiting at most ``batch_wait_ms`` for more) and runs
them as one batch, which is where nano's batched throughput comes from. A request that waits
longer than ``request_timeout_s`` gets 504. Every setting is an environment variable (see
``Settings``) or a ``opendecider serve`` flag.
"""

import asyncio
import hmac
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field, fields

log = logging.getLogger("opendecider.serve")
# a client's x-request-id is echoed and logged only if it looks like one; otherwise the server makes its own
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def _env(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not a valid {cast.__name__}") from None


@dataclass
class Settings:
    """Server settings; each field reads OPENDECIDER_<FIELD NAME IN CAPITALS> from the environment."""
    model: str = "manjunathshiva/opendecider-nano"
    revision: str = ""                # Hub revision (tag, branch or commit) to pin; empty = latest
    device: str = ""                  # empty = auto (cuda, then mps, then cpu)
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""                 # if set, require "Authorization: Bearer <api_key>"
    max_in_flight: int = 256          # admitted requests at once; excess -> 503 + Retry-After
    max_batch: int = 32               # questions per inference batch
    batch_wait_ms: float = 2.0        # how long the batcher waits to fill a batch
    request_timeout_s: float = 30.0   # admitted but unanswered after this -> 504
    max_body_bytes: int = 1_048_576   # larger request bodies -> 413
    max_questions: int = 64           # per state
    max_options: int = 256            # per question
    max_state_chars: int = 200_000    # JSON-serialised state
    max_batch_states: int = 256       # states per /v1/systemone/batch request
    threads: int = 0                  # torch intra-op threads on CPU (0 = torch default)
    dtype: str = ""                   # nano: float32 (default) or bfloat16
    small_batch: int = 1              # Qwen-based models: questions per forward pass (1 = exact published path)
    root_path: str = ""               # public URL prefix behind a reverse proxy
    log_level: str = "info"

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        vals = {}
        for f in fields(cls):
            cast = {int: int, float: float, str: str}[type(f.default)]
            vals[f.name] = _env("OPENDECIDER_" + f.name.upper(), f.default, cast)
        vals.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**vals)


class _Metrics:
    """A few counters and histograms, rendered in the Prometheus text format (no dependency)."""
    BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
    BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)

    def __init__(self):
        self.lock = threading.Lock()
        self.requests: dict[tuple, int] = {}
        self.latency = [0] * (len(self.BUCKETS) + 1)
        self.latency_sum = 0.0
        self.batch = [0] * (len(self.BATCH_BUCKETS) + 1)
        self.batch_sum = 0
        self.questions = 0
        self.in_flight = 0

    @staticmethod
    def _bucket(bounds, v):
        for i, b in enumerate(bounds):
            if v <= b:
                return i
        return len(bounds)

    def request(self, endpoint: str, code: int, seconds: float):
        with self.lock:
            self.requests[(endpoint, code)] = self.requests.get((endpoint, code), 0) + 1
            self.latency[self._bucket(self.BUCKETS, seconds)] += 1
            self.latency_sum += seconds

    def batch_run(self, n_questions: int):
        with self.lock:
            self.batch[self._bucket(self.BATCH_BUCKETS, n_questions)] += 1
            self.batch_sum += n_questions
            self.questions += n_questions

    def render(self, queue_depth: int) -> str:
        with self.lock:
            out = ["# TYPE opendecider_requests_total counter"]
            for (ep, code), n in sorted(self.requests.items()):
                out.append(f'opendecider_requests_total{{endpoint="{ep}",code="{code}"}} {n}')
            for name, bounds, counts, total in (
                    ("opendecider_request_seconds", self.BUCKETS, self.latency, self.latency_sum),
                    ("opendecider_batch_questions", self.BATCH_BUCKETS, self.batch, self.batch_sum)):
                out.append(f"# TYPE {name} histogram")
                acc = 0
                for b, c in zip(bounds, counts):
                    acc += c
                    out.append(f'{name}_bucket{{le="{b}"}} {acc}')
                acc += counts[-1]
                out += [f'{name}_bucket{{le="+Inf"}} {acc}', f"{name}_sum {total}", f"{name}_count {acc}"]
            out += ["# TYPE opendecider_questions_total counter", f"opendecider_questions_total {self.questions}",
                    "# TYPE opendecider_in_flight gauge", f"opendecider_in_flight {self.in_flight}",
                    "# TYPE opendecider_queue_depth gauge", f"opendecider_queue_depth {queue_depth}"]
        return "\n".join(out) + "\n"


@dataclass
class _Job:
    items: list
    future: Future = field(default_factory=Future)
    request_id: str = ""   # for the log if inference fails


class Batcher:
    """One inference thread; merges the questions of concurrent requests into shared batches."""

    def __init__(self, model, max_batch: int = 32, wait_ms: float = 2.0, metrics: _Metrics | None = None):
        self.model, self.max_batch, self.wait = model, max(1, max_batch), max(0.0, wait_ms) / 1000
        self.metrics = metrics
        self.q: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="opendecider-infer", daemon=True)
        self.thread.start()

    def submit(self, items: list, request_id: str = "") -> Future:
        job = _Job(items, request_id=request_id)
        self.q.put(job)
        return job.future

    def depth(self) -> int:
        return self.q.qsize()

    def _loop(self):
        while True:
            jobs = [self.q.get()]
            n = len(jobs[0].items)
            deadline = time.monotonic() + self.wait
            while n < self.max_batch:
                left = deadline - time.monotonic()
                try:
                    job = self.q.get(timeout=left) if left > 0 else self.q.get_nowait()
                except queue.Empty:
                    break
                jobs.append(job)
                n += len(job.items)
            jobs = [j for j in jobs if j.future.set_running_or_notify_cancel()]
            if jobs:
                self._run(jobs)

    def _decide(self, items):
        probs, info = [], []
        for i in range(0, len(items), self.max_batch):   # a request larger than max_batch runs in chunks
            p, inf = self.model.decide(items[i:i + self.max_batch])
            probs += p
            info += inf
            if self.metrics:
                self.metrics.batch_run(len(items[i:i + self.max_batch]))
        return probs, info

    def _run(self, jobs):
        try:
            probs, info = self._decide([it for j in jobs for it in j.items])
        except Exception as e:   # noqa: BLE001 -- isolate: one bad request must not fail the others in its batch
            if len(jobs) == 1:
                if isinstance(e, ValueError):   # the request's own fault (e.g. longer than the model's context): 422
                    jobs[0].future.set_exception(e)
                else:                           # anything else: 500 without internals; traceback in the log
                    log.exception("inference failed (request %s)", jobs[0].request_id or "-")
                    jobs[0].future.set_exception(_InferenceError())
                return
            # a merged batch failed: retry each request on its own. Log it, so a failure that only happens at this batch
            # size (e.g. out of memory) is visible even when every retry succeeds.
            log.warning("a batch of %d requests failed (%s); retrying each request on its own",
                        len(jobs), type(e).__name__)
            for j in jobs:
                self._run([j])
            return
        i = 0
        for j in jobs:
            k = len(j.items)
            j.future.set_result((probs[i:i + k], info[i:i + k]))
            i += k


class _InferenceError(RuntimeError):
    pass


def _jev_wire(result: dict) -> dict:
    """Jev's protocol: a score answer's ``score`` is the expected score (it may fall between levels).
    The library's ``score`` is the most likely level; on the wire that becomes ``level``."""
    for a in result["answers"].values():
        if a.get("type") == "score":
            a["level"] = a["score"]
            a["score"] = a.pop("expected")
    return result


def _json_size_ok(obj, limit: int) -> bool:
    return len(json.dumps(obj, ensure_ascii=False, default=str)) <= limit


def create_app(model=None, settings: Settings | None = None):
    """Build the FastAPI app. Pass a loaded model (tests, embedding); otherwise it is loaded from settings."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse

    s = settings or Settings.from_env()
    if s.threads > 0:
        import torch
        torch.set_num_threads(s.threads)
    if model is None:
        from . import load
        t0 = time.perf_counter()
        model = load(s.model, device=s.device or None, revision=s.revision or None, dtype=s.dtype or None)
        log.info("loaded %s in %.1f s", s.model, time.perf_counter() - t0)
    if hasattr(model.impl, "ping") and getattr(model.impl, "timeout", 0) > s.request_timeout_s:
        # a model served elsewhere: give up on the upstream when the client gets its 504, so a hung server does not
        # hold the inference thread (and every request behind it) for longer
        model.impl.timeout = s.request_timeout_s
    if s.small_batch > 1 and hasattr(model.impl, "batch"):
        model.impl.batch = s.small_batch
    metrics = _Metrics()
    batcher = Batcher(model, s.max_batch, s.batch_wait_ms, metrics)
    expected = ("Bearer " + s.api_key).encode("utf-8", "surrogateescape") if s.api_key else b""
    admitted = 0   # only touched on the event loop, so no lock is needed

    app = FastAPI(title="opendecider-serve", root_path=s.root_path,
                  summary="OpenDecider decisions over the TypeSafe Jev /v1/systemone protocol")

    def check_auth(request: Request):
        if not s.api_key:
            return
        supplied = (request.headers.get("authorization") or "").encode("utf-8", "surrogateescape")
        if not hmac.compare_digest(supplied, expected):   # constant time; bytes so any header is safe
            raise HTTPException(401, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    async def read_body(request: Request) -> dict:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > s.max_body_bytes:
            raise HTTPException(413, f"request body larger than {s.max_body_bytes} bytes")
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > s.max_body_bytes:
                raise HTTPException(413, f"request body larger than {s.max_body_bytes} bytes")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "request body is not valid JSON") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "request body must be a JSON object")
        return body

    def validate(questions) -> dict:
        try:
            qs = model.prepare(questions)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        if len(qs) > s.max_questions:
            raise HTTPException(422, f"at most {s.max_questions} questions per state (got {len(qs)})")
        for k, q in qs.items():
            n = len(q["criteria"]) if q["type"] != "noul" else 2
            if n > s.max_options:
                raise HTTPException(422, f"question {k!r}: at most {s.max_options} options (got {n})")
        return qs

    def check_state(state):
        if not _json_size_ok(state, s.max_state_chars):   # null is allowed, as in Jev's protocol
            raise HTTPException(422, f"'state' is longer than {s.max_state_chars} characters")

    async def infer(items: list, rid: str):
        fut = batcher.submit(items, rid)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(fut), timeout=s.request_timeout_s)
        except asyncio.TimeoutError:
            fut.cancel()   # if still queued it is skipped; if already running its result is discarded
            raise HTTPException(504, f"no answer within {s.request_timeout_s:g} s") from None
        except _InferenceError:
            raise HTTPException(500, "inference failed") from None
        except ValueError as e:   # e.g. an input longer than the model's context
            raise HTTPException(422, str(e)) from None

    async def admitted_call(request: Request, endpoint: str, handler):
        nonlocal admitted
        t0 = time.perf_counter()
        code = 500
        try:
            check_auth(request)
            if admitted >= s.max_in_flight:
                raise HTTPException(503, "server busy, try again shortly", headers={"Retry-After": "1"})
            admitted += 1
            metrics.in_flight = admitted
            try:
                body = await read_body(request)
                result = await handler(body, request.state.request_id)
            finally:
                admitted -= 1
                metrics.in_flight = admitted
            ms = (time.perf_counter() - t0) * 1000
            code = 200
            return JSONResponse(result, headers={"Server-Timing": f"total;dur={ms:.1f}"})
        except HTTPException as e:
            code = e.status_code
            raise
        finally:
            metrics.request(endpoint, code, time.perf_counter() - t0)

    async def one(body: dict, rid: str) -> dict:
        qs = validate(body.get("questions"))
        state = body.get("state")
        if state is None:
            state = ""
        check_state(state)
        probs, info = await infer(model.items(state, qs), rid)
        return _jev_wire(model.assemble(qs, probs, info))

    async def batch(body: dict, rid: str) -> dict:
        states = body.get("states")
        if not isinstance(states, list) or not states:
            raise HTTPException(422, "'states' must be a non-empty list")
        if len(states) > s.max_batch_states:
            raise HTTPException(422, f"at most {s.max_batch_states} states per batch (got {len(states)})")
        qs = validate(body.get("questions"))
        states = ["" if st is None else st for st in states]
        for st in states:
            check_state(st)
        items = [it for st in states for it in model.items(st, qs)]
        probs, info = await infer(items, rid)
        n = len(qs)
        results = [_jev_wire(model.assemble(qs, probs[i * n:(i + 1) * n], info[i * n:(i + 1) * n]))
                   for i in range(len(states))]
        return {"model": model.name, "results": results,
                "total_usage": {"input_tokens": sum(r["usage"]["input_tokens"] for r in results), "output_tokens": 0}}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        return await admitted_call(request, "systemone", one)

    @app.post("/v1/systemone/batch")
    async def systemone_batch(request: Request):
        return await admitted_call(request, "systemone_batch", batch)

    @app.get("/v1/models")
    def models():
        m = model.meta
        return {"models": [{"name": model.name, "description": m.get("description") or f"OpenDecider {m.get('kind', '')} decision model".replace("  ", " "),
                            "release_date": m.get("release_date", ""), "kind": m.get("kind")}]}

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or ""
        if not _REQUEST_ID.fullmatch(rid):
            rid = uuid.uuid4().hex
        request.state.request_id = rid
        try:
            response = await call_next(request)
        except Exception:   # noqa: BLE001 -- an unexpected error still gets the JSON body and the request id
            # %r: the path is decoded from the URL, so it could hold control characters; the id is already validated
            log.exception("unhandled error on %s %r (request %s)", request.method, request.url.path, rid)
            response = JSONResponse({"detail": "internal server error"}, status_code=500)
        response.headers["x-request-id"] = rid
        return response

    @app.get("/health")
    def health():
        from . import __version__
        return {"status": "ok", "model": model.name, "kind": model.meta.get("kind"),
                "device": str(getattr(model.impl, "device", "unknown")), "version": __version__,
                "in_flight": admitted, "queue_depth": batcher.depth()}

    @app.get("/ready")
    def ready():
        alive = batcher.thread.is_alive()
        ping = getattr(model.impl, "ping", None)   # a model served elsewhere (LM Studio / Ollama) must be reachable
        upstream = ping() if ping else None
        ok = alive and upstream is not False
        body = {"ready": ok} if upstream is None else {"ready": ok, "upstream": upstream}
        return body if ok else JSONResponse(body, status_code=503)

    @app.get("/metrics")
    def prometheus():
        return PlainTextResponse(metrics.render(batcher.depth()), media_type="text/plain; version=0.0.4")

    app.state.model, app.state.settings, app.state.batcher, app.state.metrics = model, s, batcher, metrics
    return app


def run(settings: Settings) -> None:
    import uvicorn
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(settings=settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
