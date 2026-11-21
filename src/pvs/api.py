"""FastAPI service: still detection, health/readiness, metadata, live results, preview, metrics.

Defaults: bind loopback; no auth needed on loopback. For shared deployments set
``api_token_env`` in the station config (bearer token) and terminate TLS in
the reverse proxy (deploy/Caddyfile).
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import threading
import time
from datetime import UTC
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .capture import DecodeError, Frame, decode_image_bytes
from .config import StationConfig
from .schema import ErrorResponse, FrameResult, FrameStatus

REQS = Counter("pvs_requests_total", "detect requests", ["status"])
STAGE = Histogram("pvs_stage_ms", "stage time ms", ["stage"], buckets=(5, 10, 25, 50, 100, 150, 250, 500, 1000, 5000, 20000))
LIVE_PROCESSED = Counter("pvs_live_processed_total", "live frames processed")
LIVE_SCHEDULED = Counter("pvs_live_scheduled_total", "live samples scheduled")
LIVE_DROPPED = Gauge("pvs_live_dropped_frames", "frames discarded as stale/overflow")
IDENT = Counter("pvs_identity_total", "instances by identity level", ["level"])
MODEL_OK = Gauge("pvs_model_ready", "1 if models loaded")
QUEUE_AGE = Histogram("pvs_frame_age_ms", "capture-to-result ms (live)", buckets=(25, 50, 100, 150, 250, 500, 1000, 5000))


def _record(r: FrameResult) -> None:
    REQS.labels(r.status.value).inc()
    for k, v in r.timing_ms.items():
        STAGE.labels(k).observe(v)
    for i in r.instances:
        IDENT.labels(i.identity.level.value).inc()


class LiveWorker:
    """One worker per stream: replay file or camera -> latest result + overlay."""

    def __init__(self, pipeline, cfg: StationConfig, video: str | None, device: str | None) -> None:
        from .tracker import Tracker

        self.p, self.cfg = pipeline, cfg
        self.video, self.device = video, device
        self.tracker = Tracker(cfg.camera_id)
        self.latest: FrameResult | None = None
        self.latest_jpg: bytes | None = None
        self.stale = True
        self._stop = threading.Event()
        self._t = threading.Thread(target=self.run, daemon=True)
        self.cond = threading.Condition()

    def start(self) -> None:
        self._t.start()

    def stop(self) -> None:
        self._stop.set()

    def _publish(self, f: Frame, r: FrameResult) -> None:
        from .render import draw

        ok, jpg = cv2.imencode(".jpg", draw(f.image, r, self.cfg.capture.mirror_preview, self.stale))
        with self.cond:
            self.latest, self.latest_jpg = r, jpg.tobytes() if ok else None
            self.cond.notify_all()
        _record(r)

    def run(self) -> None:
        period = 1.0 / self.cfg.capture.sample_hz
        if self.video:
            from .capture import replay

            loop = 0
            while not self._stop.is_set():
                for f in replay(self.video, self.cfg.camera_id, self.cfg.capture.sample_hz):
                    if self._stop.is_set():
                        return
                    f.epoch = loop  # each pass over the file is a new stream epoch: tracks/evidence reset
                    t0 = time.monotonic()
                    LIVE_SCHEDULED.inc()
                    self.stale = False
                    self._publish(f, self.p.process(f, self.tracker))
                    LIVE_PROCESSED.inc()
                    time.sleep(max(0.0, period - (time.monotonic() - t0)))
                loop += 1
        else:
            from .capture import LiveCamera

            c = self.cfg.capture
            cam = LiveCamera(self.device or c.device, self.cfg.camera_id, c.width, c.height, c.queue_size,
                             c.max_frame_age_ms, c.reconnect_s)
            cam.start()
            while not self._stop.is_set():
                t0 = time.monotonic()
                LIVE_SCHEDULED.inc()
                f = cam.latest()
                self.stale = cam.stale(c.stall_timeout_s)
                LIVE_DROPPED.set(cam.dropped_stale + cam.dropped_overflow)
                if f is not None:
                    r = self.p.process(f, self.tracker)
                    QUEUE_AGE.observe((time.monotonic() - f.mono_t) * 1000)
                    self._publish(f, r)
                    LIVE_PROCESSED.inc()
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
            cam.stop()


def create_app(pipeline, cfg: StationConfig, video: str | None = None, device: str | None = None) -> FastAPI:
    app = FastAPI(title="Product Vision Station", version="0.1.0")
    sem = asyncio.Semaphore(cfg.limits.max_concurrency)
    token = os.environ.get(cfg.api_token_env) if cfg.api_token_env else None
    MODEL_OK.set(0 if pipeline.failed else 1)
    worker = LiveWorker(pipeline, cfg, video, device) if (video or device) else None
    if worker:
        worker.start()

    def auth(authorization: str | None = Header(default=None)) -> None:
        if token and not (authorization and hmac.compare_digest(authorization, f"Bearer {token}")):
            raise HTTPException(401, "unauthorised")

    def err(code: int, status: FrameStatus, e: str, d: str) -> JSONResponse:
        return JSONResponse(ErrorResponse(status=status, error=e, detail=d).model_dump(), status_code=code)

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    @app.get("/readyz")
    def ready():
        if pipeline.failed:
            return JSONResponse({"ready": False, "status": "model_failed", "reason": pipeline.failed}, 503)
        return {"ready": True, "profile": pipeline.profile.value, "bundle": cfg.bundle_id}

    @app.get("/v1/meta", dependencies=[Depends(auth)])
    def meta():
        return {"profile": pipeline.profile.value, "backend": pipeline.backend, "bundle": cfg.bundle_id,
                "catalogue_revision": pipeline.catalogue["revision"], "thresholds": pipeline.thr.version,
                "schema_version": "1.0", "limits": cfg.limits.model_dump(),
                "products": [{"sku": p["sku"], "family": p["family"], "approval_state": p["approval_state"]}
                             for p in pipeline.catalogue["products"]]}

    @app.get("/v1/schema")
    def schema():
        return FrameResult.model_json_schema()

    @app.post("/v1/detect", response_model=FrameResult, dependencies=[Depends(auth)],
              responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
                         503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
    async def detect(file: UploadFile = File(...)):
        data = await file.read(cfg.limits.max_upload_bytes + 1)
        if len(data) > cfg.limits.max_upload_bytes:
            return err(413, FrameStatus.DECODE_FAILED, "payload_too_large", f"max {cfg.limits.max_upload_bytes} bytes")
        if not data:
            return err(400, FrameStatus.DECODE_FAILED, "empty_upload", "no bytes received")
        try:
            img, T, _ = decode_image_bytes(data, cfg.limits.max_pixels, tuple(cfg.limits.formats))
        except DecodeError as e:
            code = 415 if str(e).startswith("unsupported_format") else (413 if "too_many_pixels" in str(e) else 400)
            return err(code, FrameStatus.DECODE_FAILED, str(e).split(":")[0], str(e))
        import hashlib
        from datetime import datetime

        f = Frame(img, cfg.camera_id, 0, 0, datetime.now(UTC), time.monotonic(),
                  f"upload:{file.filename}", file.filename, T, hashlib.sha256(data).hexdigest())
        if pipeline.failed:
            r = pipeline.process(f)
            _record(r)
            return JSONResponse(json.loads(r.model_dump_json()), status_code=503)
        async with sem:  # bounded concurrency; work past its deadline is discarded
            try:
                r = await asyncio.wait_for(asyncio.to_thread(pipeline.process, f), cfg.limits.request_deadline_s)
            except TimeoutError:
                return err(504, FrameStatus.MODEL_FAILED, "deadline_exceeded", f">{cfg.limits.request_deadline_s}s")
        _record(r)
        return JSONResponse(json.loads(r.model_dump_json()))

    @app.get("/v1/live/latest", dependencies=[Depends(auth)])
    def live_latest():
        if not worker or worker.latest is None:
            return JSONResponse({"status": "no_live_source" if not worker else "warming_up"}, 404)
        return json.loads(worker.latest.model_dump_json()) | {"camera_stale": worker.stale}

    @app.get("/v1/live/latest.jpg", dependencies=[Depends(auth)])
    def live_jpg():
        if not worker or worker.latest_jpg is None:
            raise HTTPException(404, "no frame")
        return Response(worker.latest_jpg, media_type="image/jpeg")

    @app.websocket("/v1/live")
    async def live_ws(ws: WebSocket):
        if token and ws.headers.get("authorization") != f"Bearer {token}":
            await ws.close(code=4401)
            return
        await ws.accept()
        last = None
        try:
            while True:
                if worker and worker.latest is not None and worker.latest is not last:
                    last = worker.latest
                    await ws.send_text(last.model_dump_json())
                await asyncio.sleep(0.03)
        except WebSocketDisconnect:
            return

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", response_class=HTMLResponse)
    def preview():
        return (Path(__file__).parent / "preview.html").read_text()

    app.state.worker = worker
    return app
