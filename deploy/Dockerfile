# Station runtime image (CPU). GPU: build with BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# and install onnxruntime-gpu instead of onnxruntime; verify providers with `pvs doctor`.
# NOTE: this Dockerfile has not been built in the development sandbox (no Docker daemon); see docs/VERIFICATION.md.
ARG BASE=python:3.12-slim
FROM ${BASE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg tini \
    && rm -rf /var/lib/apt/lists/* && useradd -m -u 10001 pvs
WORKDIR /app
COPY requirements/runtime.lock requirements/runtime.lock
RUN pip install -r requirements/runtime.lock
ARG WITH_MODELS=1
COPY requirements/models.lock requirements/models.lock
RUN if [ "$WITH_MODELS" = "1" ]; then pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements/models.lock; fi
COPY pyproject.toml README.md ./
COPY src src
COPY configs configs
COPY catalogues catalogues
COPY models/manifest.json models/manifest.json
COPY scripts scripts
COPY deploy/trust deploy/trust
RUN pip install --no-deps . && mkdir -p /app/var && chown -R pvs /app/var
USER pvs
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
ENTRYPOINT ["tini", "--"]
CMD ["pvs", "--profile", "station", "--config", "/app/configs/station.pretrained.yaml", "serve", "--host", "0.0.0.0"]
