# OpenDecider server, CPU. Build from the repository root:
#   docker build -f docker/Dockerfile -t opendecider:cpu .
#   docker run -p 8000:8000 -v opendecider-models:/models opendecider:cpu
# Bake a model into the image instead of downloading it at first start:
#   docker build -f docker/Dockerfile --build-arg BAKE_MODEL=manjunathshiva/opendecider-nano -t opendecider:nano .
# The base image is pinned by digest and every package by hash (docker/requirements-cpu.txt, from docker/lock.sh);
# Dependabot updates the base image's digest; refresh the packages with `bash docker/lock.sh`.
FROM python:3.14-slim@sha256:0741d101873c12ab927e6f8653feb8862b9bd58771177acb1b885b95141f91b4

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models OPENDECIDER_HOST=0.0.0.0 OPENDECIDER_PORT=8000

WORKDIR /app
COPY docker/requirements-cpu.txt ./
RUN pip install --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-cpu.txt
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY opendecider ./opendecider
RUN pip install --no-deps --no-build-isolation . && pip check

RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
USER app

ARG BAKE_MODEL=""
RUN if [ -n "$BAKE_MODEL" ]; then python -c "from huggingface_hub import snapshot_download; snapshot_download('$BAKE_MODEL')"; fi
ENV OPENDECIDER_MODEL=${BAKE_MODEL:-manjunathshiva/opendecider-nano}

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"
ENTRYPOINT ["opendecider", "serve"]
