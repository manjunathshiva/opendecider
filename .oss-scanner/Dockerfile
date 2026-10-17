# OpenDecider for Anthropic's OSS Scanner (https://github.com/anthropics/oss-scanner): the Python package (library,
# `opendecider serve`, `opendecider mcp`, opendecider.integrations), @opendecider/client (typescript/),
# @opendecider/web (web/) and the Chrome extension (extension/), built in /src with their tests runnable offline.
# Mirrors .github/workflows/ci.yml, typescript.yml, web.yml and extension.yml. Build from the repository root:
#   docker build -f .oss-scanner/Dockerfile -t opendecider:oss-scanner .
# CPU only. No model weights: the Python tests use a fake model and the web tests web/test/fixtures/tiny.onnx.
# The base image is docker/Dockerfile's, pinned by digest; Node and uv are copied from images pinned the same way.
FROM node:24-trixie-slim@sha256:7f486cabe892bff8fef04c0f7da8dba84a560be3e4e2740b7a86afc57ebec315 AS node
FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv
FROM python:3.14-slim@sha256:a2b82f3c48559aa0a8446d9af49826b6e2b2016f4cd2afabfe6013ec53729170

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    npm_config_update_notifier=false \
    AGNO_TELEMETRY=false CREWAI_DISABLE_TELEMETRY=true CREWAI_TRACING_ENABLED=false OTEL_SDK_DISABLED=true \
    MASTRA_TELEMETRY_DISABLED=1 UV_NO_CACHE=1 UV_PYTHON_INSTALL_DIR=/opt/uv-python

# Node 24 LTS, as extension.yml
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version && npm --version

# The server's dependencies as docker/Dockerfile installs them: by hash, torch from PyTorch's CPU index
COPY docker/requirements-cpu.txt /tmp/requirements-cpu.txt
RUN pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-cpu.txt
# The test dependencies of ci.yml's test job (MCP 2, the agent frameworks; unpinned there on purpose). uv: pip takes
# minutes backtracking through google-adk's dependencies on top of the locked set.
COPY --from=uv /uv /usr/local/bin/uv
RUN uv pip install --system pytest fastapi httpx "mcp>=2.2,<3" langchain-core langgraph llama-index-core agno \
      agent-framework-core google-adk pydantic-ai-slim opentelemetry-sdk uvicorn
# ci.yml's frameworks-mcp1 job: crewai and strands-agents pin mcp 1.x, so they get their own venv, torch-free, on
# Python 3.12 as in CI (crewai's numpy pin has no 3.14 wheels)
RUN uv venv --python 3.12 /opt/mcp1 \
    && uv pip install --python /opt/mcp1/bin/python pytest crewai strands-agents opentelemetry-sdk

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# The package, editable so a patch to opendecider/ takes effect at once
RUN pip install --no-deps --no-build-isolation -e . \
    && uv pip install --python /opt/mcp1/bin/python --no-deps -e . \
    && opendecider --help >/dev/null
# @opendecider/client, then @opendecider/web (built on this checkout's client), then the extension (built on both).
# --ignore-scripts as in CI: no dependency runs install scripts. Playwright's Chromium (the extension's e2e) is left
# out: its test also needs the 594 MB q8 model.
RUN cd typescript && npm ci --ignore-scripts --no-audit --no-fund && npm run build
RUN cd web && npm ci --ignore-scripts --no-audit --no-fund && npm run build
RUN cd extension && npm ci --ignore-scripts --no-audit --no-fund && npm run build

# web.yml's parity check reads opendecider-nano's tokenizer (3.5 MB) at its pinned revision: fetched now, so the
# check also runs offline from the Hub cache
RUN cd web && (PYTHONPATH=.. python test/fixtures/make_fixtures.py --check \
      || echo "WARNING: web fixtures check reported failures")

# Run the cheap tests so the image is known to work; failures do not abort the build.
RUN (python -m pytest -q -p no:cacheprovider tests || echo "WARNING: pytest reported failures") \
    && (/opt/mcp1/bin/python -m pytest -q -p no:cacheprovider tests/test_frameworks.py tests/test_decisions.py \
          tests/test_guard.py tests/test_guard_frameworks.py || echo "WARNING: pytest (mcp1) reported failures") \
    && (cd typescript && PYTHONPATH=.. python test/fixtures/make_fixtures.py --check \
          || echo "WARNING: typescript fixtures check reported failures") \
    && (cd typescript && npm run typecheck && npm test || echo "WARNING: typescript tests reported failures") \
    && (cd web && npm run typecheck && npm test || echo "WARNING: web tests reported failures") \
    && (cd extension && npm run typecheck && npm test || echo "WARNING: extension tests reported failures")

# The scan runs offline: the Hub client fails at once instead of waiting on the network
ENV HF_HUB_OFFLINE=1
