"""Frame intake: still images, prerecorded replay and live cameras share one Frame type.

Capture time and sequence are read at intake. EXIF orientation is applied
exactly once here; downstream code only sees the declared upright source frame.
"""
from __future__ import annotations

import io
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .transforms import Transform, apply_exif


class DecodeError(ValueError):
    pass


@dataclass
class Frame:
    image: np.ndarray  # BGR, upright, unmirrored source frame
    camera: str
    epoch: int
    frame_id: int
    captured_at: datetime
    mono_t: float  # monotonic seconds at intake (live) or media timestamp (replay)
    source: str
    source_key: str | None = None
    raw_to_source: Transform = field(default_factory=Transform)
    content_hash: str | None = None

    @property
    def size_wh(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


def decode_image_bytes(data: bytes, max_pixels: int = 40_000_000, formats: tuple[str, ...] | None = None
                       ) -> tuple[np.ndarray, Transform, str]:
    """Decode bytes with PIL (format sniffing + EXIF), enforce pixel limits."""
    from PIL import Image, UnidentifiedImageError

    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        im = Image.open(io.BytesIO(data))
        fmt = (im.format or "").lower()
        if formats and fmt not in formats:
            raise DecodeError(f"unsupported_format:{fmt}")
        if im.width * im.height > max_pixels:
            raise DecodeError("too_many_pixels")
        orientation = im.getexif().get(0x0112, 1)
        rgb = np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise DecodeError(f"undecodable:{type(e).__name__}") from e
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    up, T = apply_exif(bgr, orientation)
    return up, T, fmt


def frame_from_file(path: str | Path, camera: str = "file", frame_id: int = 0, max_pixels: int = 40_000_000) -> Frame:
    import hashlib

    p = Path(path)
    data = p.read_bytes()
    img, T, _ = decode_image_bytes(data, max_pixels)
    return Frame(img, camera, 0, frame_id, datetime.fromtimestamp(p.stat().st_mtime, UTC), 0.0,
                 str(p), p.name, T, hashlib.sha256(data).hexdigest())


def replay(path: str | Path, camera: str = "replay", sample_hz: float | None = None,
           start: datetime | None = None) -> Iterator[Frame]:
    """Replay a video file in source order, exposing original media timestamps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DecodeError(f"cannot_open:{path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    period = 1.0 / sample_hz if sample_hz else 0.0
    next_t = 0.0
    idx = -1
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            idx += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= 0 and idx > 0:
                t = idx / fps
            if period and t + 1e-6 < next_t:
                continue
            next_t = t + period if period else 0
            yield Frame(img, camera, 0, idx, start + timedelta(seconds=t), t, str(path), f"{Path(path).name}#{idx}")
    finally:
        cap.release()


class LiveCamera:
    """Bounded latest-frame capture with stale-frame discard and reconnect epochs."""

    def __init__(self, device: str | int, camera: str, width: int, height: int, queue_size: int = 2,
                 max_age_ms: float = 250.0, reconnect_s: float = 2.0) -> None:
        self.device = int(device) if str(device).isdigit() else device
        self.camera = camera
        self.width, self.height = width, height
        self.q: deque[Frame] = deque(maxlen=queue_size)
        self.max_age_ms = max_age_ms
        self.reconnect_s = reconnect_s
        self.epoch = 0
        self.seq = 0
        self.dropped_overflow = 0
        self.dropped_stale = 0
        self.resets = 0
        self.last_frame_mono: float | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.resolution: tuple[int, int] | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        cap = self._open()
        while not self._stop.is_set():
            ok, img = cap.read() if cap.isOpened() else (False, None)
            if not ok:
                cap.release()
                with self._lock:
                    self.epoch += 1
                    self.resets += 1
                    self.q.clear()
                time.sleep(self.reconnect_s)
                cap = self._open()
                continue
            now = time.monotonic()
            res = (img.shape[1], img.shape[0])
            with self._lock:
                if self.resolution and res != self.resolution:
                    self.epoch += 1  # resolution change starts a new epoch
                    self.q.clear()
                self.resolution = res
                if len(self.q) == self.q.maxlen:
                    self.dropped_overflow += 1
                self.q.append(Frame(img, self.camera, self.epoch, self.seq, datetime.now(UTC), now,
                                    f"camera:{self.device}"))
                self.seq += 1
                self.last_frame_mono = now
        cap.release()

    def latest(self) -> Frame | None:
        """Newest frame; older queued frames are discarded as stale."""
        with self._lock:
            if not self.q:
                return None
            f = self.q.pop()
            self.dropped_stale += len(self.q)
            self.q.clear()
        if (time.monotonic() - f.mono_t) * 1000 > self.max_age_ms:
            self.dropped_stale += 1
            return None
        return f

    def stale(self, timeout_s: float) -> bool:
        return self.last_frame_mono is None or time.monotonic() - self.last_frame_mono > timeout_s


def frame_quality(img: np.ndarray) -> dict:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (min(640, g.shape[1]), int(g.shape[0] * min(640, g.shape[1]) / g.shape[1])))
    return {"mean_luma": float(g.mean()), "saturated_frac": float((g >= 250).mean()),
            "sharpness": float(cv2.Laplacian(small, cv2.CV_64F).var())}
