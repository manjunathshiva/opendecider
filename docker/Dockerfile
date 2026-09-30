# OpenDecider server, CPU. Build from the repository root:
#   docker build -f docker/Dockerfile -t opendecider:cpu .
#   docker run -p 8000:8000 -v opendecider-models:/models opendecider:cpu
# Bake a model into the image instead of downloading it at first start:
#   docker build -f docker/Dockerfile --build-arg BAKE_MODEL=manjunathshiva/opendecider-nano -t opendecider:nano .
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models OPENDECIDER_HOST=0.0.0.0 OPENDECIDER_PORT=8000

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY opendecider ./opendecider
RUN pip install ".[serve,small]"

RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
USER app

ARG BAKE_MODEL=""
RUN if [ -n "$BAKE_MODEL" ]; then python -c "from huggingface_hub import snapshot_download; snapshot_download('$BAKE_MODEL')"; fi
ENV OPENDECIDER_MODEL=${BAKE_MODEL:-manjunathshiva/opendecider-nano}

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"
ENTRYPOINT ["opendecider", "serve"]
