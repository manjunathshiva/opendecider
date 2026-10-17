# Threat model

## What this project does and where untrusted input enters

OpenDecider is a family of small decision models and the code that runs them: typed questions (`choice`, `score`,
`noul`) about a state (text or JSON), answered with a probability for every option. Nothing in a state is executed;
it is tokenised and scored. The parts, and where untrusted input enters each:

- **`opendecider serve`** (`opendecider/serve.py`, FastAPI + uvicorn): an HTTP server speaking TypeSafe Jev's
  `/v1/systemone` protocol. **Untrusted: every request.** It binds `0.0.0.0:8000` by default and has no API key unless
  `OPENDECIDER_API_KEY` is set (`Authorization: Bearer`, compared with `hmac.compare_digest`). Only
  `POST /v1/systemone` and `POST /v1/systemone/batch` check the key; `GET /health`, `/ready`, `/metrics`,
  `/v1/models` and FastAPI's `/docs` and `/openapi.json` are not behind the key (health checks, Prometheus).
  Request body: `{"state", "questions"}` (batch: `{"states", "questions"}`); a `model` field is ignored, the model is
  fixed at startup by the operator. Limits (`Settings`): body 1 MiB (413), 64 questions, 256 options, 200,000 state
  characters, 256 states per batch, 256 admitted requests (503), 30 s per request (504). Errors must never return
  tracebacks or paths; `x-request-id` is echoed only if it matches `[A-Za-z0-9._:-]{1,128}`.
- **`opendecider mcp`** (`opendecider/mcp_server.py`, shared core `opendecider/tools.py`): an MCP server over stdio
  only. **Untrusted: tool arguments**, written by an AI agent that may itself be steered by prompt injection. Tools:
  `decide`, `choose`, `yes_no`, `score`, `decide_batch`, `guard`, `status`; the same limits as the server
  (`tools.check`, `MAX_BATCH_ITEMS` 1024). No tool loads a model by name, reads files or makes network requests of
  the caller's choosing; the model is fixed by `--model` / `OPENDECIDER_MODEL`.
- **Model loading** (`opendecider/__init__.py` `load`, `nano.py`, `small.py`, `mlx_small.py`): a Hub name or a local
  folder chosen by the operator or the calling program (never by a request). A folder with `opendecider.json` is used
  as is; anything else goes to `huggingface_hub.snapshot_download`. `opendecider.json` picks the loader (`kind`) and,
  for `small` / `small-mlx`, names a second repository to download (`base_model`, `mlx_base`), at its latest
  revision: `revision` pins only the first. Weights load through `transformers` `AutoModel` /
  `AutoModelForCausalLM` / `AutoTokenizer.from_pretrained`, `peft` `PeftModel.from_pretrained`, `mlx_lm.load` and
  `safetensors.torch.load_file` (nano's `head.safetensors`). OpenDecider never passes `trust_remote_code`
  (transformers' default is off) and never calls `torch.load` or `pickle` itself. It does not force
  `use_safetensors=True`: a repository that ships only `pytorch_model.bin` or `adapter_model.bin` is read by
  transformers' / peft's own `torch.load(weights_only=True)`. A model repository is trusted to the extent that the
  user chose it; code execution from a repository despite these settings is in scope.
- **Remote backends** (`opendecider/remote.py`): `load("ollama:…" | "lmstudio:…" | "openai:…")` and
  `load("http(s)://…")` (another `opendecider serve`). URLs come from the program or `OPENDECIDER_REMOTE_URL`; **the
  responses are untrusted** (parsed as JSON, read for probabilities). `OPENDECIDER_REMOTE_API_KEY` is sent as an
  unredirected header (never on a redirect) and over plain HTTP only to loopback unless
  `OPENDECIDER_REMOTE_ALLOW_HTTP=1`. Text shaped like a Qwen special token (`<|name|>`) is escaped before it reaches
  such a server (`prompt.for_server`).
- **Prompts** (`opendecider/prompt.py`): states and question text are encoded with `split_special_tokens=True`, so
  "[MASK]" or "<|im_end|>" in input stays text. Producing a control token from input text, or a wrong number of
  option markers, is in scope.
- **`opendecider.guard`** and **`opendecider.integrations`** (LangChain/LangGraph, LlamaIndex, Agno, CrewAI, Microsoft
  Agent Framework, Google ADK, PydanticAI, Strands): thin adapters around `tools`. Untrusted input is the text and
  tool arguments those frameworks pass in. The guard blocks on error by default (`on_error="block"`); a path that
  makes it pass text it could not check, under that default, is in scope.
- **`@opendecider/client`** (`typescript/`, no runtime dependencies): the HTTP client for the servers above
  (`src/http.ts`, `src/backends.ts`). **Untrusted: server responses.** Credentials (`apiKey` or any custom header)
  are never sent on a redirect (`redirect: "manual"` when credentials are set), never over plain HTTP to a non-loopback
  host unless `allowInsecureHttp`, and URLs with userinfo, a query or a fragment are rejected.
- **`@opendecider/web`** (`web/`): opendecider-nano in the browser / Node / Bun / Deno with `onnxruntime-web`.
  Downloads only the files pinned in `web/src/manifest.ts` (one revision of `manjunathshiva/opendecider-nano-ONNX`,
  each by size and SHA-256) from `HUB_URL` or a caller-given `baseUrl` / `files` reader. `web/src/hub.ts` refuses a
  response larger than the pinned size while streaming, checks size and SHA-256 before use, and keeps files in the
  Cache API or, where that refuses them, the origin private file system (OPFS), keyed by SHA-256 and checked again on
  every load (a mismatching cached file is deleted). Loading a file that does not match its pin is in scope.
- **OpenDecider Focus** (`extension/`, Chrome MV3): filters the YouTube feed with @opendecider/web in an offscreen
  page. Permissions: `storage`, `offscreen`, `contextMenus`, `alarms`; no host permissions; the content script runs
  only on `https://www.youtube.com/*`. **Untrusted: the YouTube page** (DOM titles and channel names read by
  `src/content.ts`) and **text the user selects on any page** (right-click "Check with OpenDecider guard").
  - `src/background.ts` (service worker) accepts messages only from its own extension id; `triage` only from a tab
    whose URL starts with `https://www.youtube.com/`; `download` and `guard` only from the extension's own pages;
    `progress` only from `offscreen.html`. `src/messages.ts` `readVideos` validates a page's list (11-character video
    ids, at most 16 videos, titles 150 and channels 60 characters). The page gets only "ok"/"hide" back.
  - `src/offscreen.ts` answers only messages with `target: "offscreen"` from its own extension id.
  - Selected text (at most 20,000 characters) goes through `chrome.storage.session` to `check.html`, rendered with
    `textContent` only. The popup (`src/popup.ts`) uses `textContent` only.
  - Nothing is downloaded before the user presses Download in the popup (`consent` in `chrome.storage.local`).
    Extension-page CSP: `script-src 'self' 'wasm-unsafe-eval'`, `connect-src 'self'` plus huggingface.co / hf.co;
    ONNX Runtime's wasm ships inside the extension.
  - Settings (`chrome.storage.sync`) are re-validated on read (`src/settings.ts` `readSettings`).
- **Docker images** (`docker/Dockerfile`, `docker/Dockerfile.cuda`, `compose.yaml`): run `opendecider serve` as uid
  10001, base image pinned by digest and Python packages by hash. Same exposure as the server.

## Components that matter most / least

- Most: `opendecider/serve.py` (network-facing by default), model loading (`__init__.py` `load`, `nano.py`,
  `small.py`, `mlx_small.py`), `mcp_server.py` and `tools.py`, `remote.py`, `prompt.py`; `web/src/hub.ts` and
  `manifest.ts` (integrity of downloaded model files); the extension's message handling (`background.ts`,
  `offscreen.ts`, `content.ts`, `messages.ts`); `typescript/src/http.ts` (credential handling).
- Less: `opendecider/integrations/` (adapters; the frameworks' own behaviour is theirs), `guard.py` thresholds,
  `bench_speed.py`, the popup and settings UI.
- Out of scope: `benchmarks/`, `examples/`, `notebooks/`, `docs/` and the documentation site, `packaging/onnx/`
  (maintainer scripts that export the ONNX builds), `typescript/compat/` and `extension/e2e/` (test-only),
  `.github/workflows/`, and bugs in third-party dependencies themselves (transformers, peft, mlx-lm, onnxruntime-web,
  @huggingface/tokenizers, FastAPI, the MCP SDK, agent frameworks) unless OpenDecider's own use of them creates the
  vulnerability. `opendecider-client` (`packaging/build_client.py`) is the same `opendecider/` source without torch:
  findings in that source count once.

## How to exercise it

- Python (installed editable in `/src`, Python 3.14, CPU torch): `python -m pytest -q tests`. The torch-free CrewAI and
  Strands tests use their own venv: `/opt/mcp1/bin/python -m pytest -q tests/test_frameworks.py tests/test_decisions.py
  tests/test_guard.py tests/test_guard_frameworks.py`.
- The server without a model download: `tests/test_serve.py` builds it with a fake model,
  `create_app(model=OpenDecider(Fake(), {"name": "test", "kind": "nano"}), settings=Settings(...))`, and drives it with
  FastAPI's `TestClient`; the same app can be served with `uvicorn.run(app, host="127.0.0.1", port=8000)` (a fake's
  `decide_many(items, info=None)` must append one `{"input_tokens", "truncated"}` per item to `info`). MCP:
  `tests/test_mcp.py` builds the server the same way (`build_server(Decider("name", loader=...))`).
- No model weights are in the image and the scan has no network (`HF_HUB_OFFLINE=1`), so `opendecider serve` /
  `opendecider mcp` with a Hub model name cannot load a real model here; use the fakes above, or a local folder with
  `opendecider.json` to exercise `load`.
- TypeScript: `cd typescript && npm test` (fake servers); `cd web && npm test` (the whole pipeline on
  `web/test/fixtures/tiny.onnx`); `cd extension && npm test` (its logic; built extension in `extension/dist/`).
  Parity with Python: `PYTHONPATH=.. python test/fixtures/make_fixtures.py --check` in `typescript/` and `web/`.
  The extension's Chromium end-to-end test (`npm run e2e`) needs Playwright's browser and the 594 MB model, so it is
  not set up.

## How you rate severity

- Critical: code execution from a request to `opendecider serve` or an MCP tool call; code execution from loading a
  model despite `trust_remote_code` being off (e.g. a path to unrestricted unpickling); authentication bypass on
  `POST /v1/systemone` when `OPENDECIDER_API_KEY` is set; @opendecider/web or the extension using a model file that
  does not match its pinned SHA-256.
- High: a request reading or writing files, or making the server fetch a URL of the requester's choosing (path
  traversal, SSRF); tracebacks, paths or the API key leaking to a client; the API key sent somewhere other than the
  configured origin (a redirect, plain HTTP to a remote host) by `remote.py` or `@opendecider/client`; a YouTube page
  or other web page making the extension do anything beyond hiding or showing its own tiles (triggering a download,
  reading storage, running the guard, script execution in an extension page); XSS in an extension page.
- Medium: an input that becomes a model control token or misplaces option markers (wrong answers that look valid);
  bypassing a size limit so one request costs far more than the limits allow; the guard passing text it could not
  check under `on_error="block"`; a crash of the server process from one request.
- Low: denial of service that needs requests within the documented limits (the server is meant to run behind a
  reverse proxy when exposed), what the endpoints outside the key (`/health`, `/ready`, `/metrics`, `/v1/models`,
  `/docs`) show, local-only issues on a developer's own machine.
- Not a vulnerability: a model answering wrongly, including the guard model missing or wrongly flagging a jailbreak or
  prompt injection (adversarial text steering a decision is a model-quality issue, documented in SECURITY.md); the
  server accepting any request when no API key is set (documented); a model repository the user explicitly chose
  serving bad answers.

## Anything to leave alone

- Accuracy, calibration and speed numbers (`benchmarks/`); disagreements between quantised builds.
- Plain HTTP on the server (TLS belongs in a reverse proxy, as SECURITY.md says) and the default bind to `0.0.0.0`
  (documented; the operator sets `--host`).
- Known advisories in optional dependencies listed in SECURITY.md (`nltk` via llama-index-core, `chromadb` via
  crewai; setuptools 81 in the CUDA image).
- Reports: please include a reproducer against `/src` (a pytest test is ideal) and, where you can, a patch that keeps
  the wire format and public API in AGENTS.md unchanged.
