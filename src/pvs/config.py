"""Machine-readable station configuration and versioned thresholds."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CaptureCfg(BaseModel):
    device: str = "file"  # "file", "/dev/video0", "rtsp://...", or a video path
    width: int = 1920
    height: int = 1080
    sample_hz: float = 15.0
    mirror_preview: bool = False
    queue_size: int = 2
    max_frame_age_ms: float = 250.0
    reconnect_s: float = 2.0
    stall_timeout_s: float = 2.0


class ModelsCfg(BaseModel):
    detector: str = "grounding_dino"  # grounding_dino | rtdetr_onnx | scripted
    masks: str = "sam2"  # sam2 | grabcut_fallback | scripted
    embedder: str = "dinov2"  # dinov2 | colour_hist
    prompts: list[str] = Field(default_factory=lambda: ["wrapped chocolate bar", "snack package"])
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    detector_max_side: int = 800
    rtdetr_onnx: str | None = None
    provider: str = "CPUExecutionProvider"


class LimitsCfg(BaseModel):
    max_upload_bytes: int = 15 * 1024 * 1024
    max_pixels: int = 40_000_000
    formats: list[str] = Field(default_factory=lambda: ["jpeg", "png", "webp", "bmp"])
    max_instances: int = 3
    request_deadline_s: float = 60.0
    max_concurrency: int = 1


class StationConfig(BaseModel):
    station_id: str = "station-01"
    camera_id: str = "station-01"
    capture: CaptureCfg = Field(default_factory=CaptureCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    limits: LimitsCfg = Field(default_factory=LimitsCfg)
    catalogue: str = "catalogues/fixture/catalogue.json"
    gallery: str | None = None  # built gallery .npz; built on demand if missing
    thresholds: str = "configs/thresholds.yaml"
    bundle_id: str = "dev-unbundled"
    calibration: dict | None = None  # measured FOV, working distance, homography when available
    outbox_path: str = "var/outbox.sqlite"
    bind: str = "127.0.0.1"
    port: int = 8000
    api_token_env: str | None = None  # set for shared deployments; loopback default needs none


class Thresholds(BaseModel):
    version: str
    sku_score: float
    sku_margin: float
    family_score: float
    tie_margin: float
    template_min_inliers: int
    template_min_spread: float
    template_max_reproj_px: float
    cue_margin: float = 0.08
    track_hold_s: float
    evidence_ttl_s: float
    axis_min_elongation: float
    axis_min_extent_px: float
    axis_max_residual: float
    smoother_window_s: float
    smoother_min_resultant: float
    quality_min_luma: float
    quality_max_saturated: float
    quality_min_sharpness: float
    calibrated: bool = False


def load_station(path: str | Path) -> StationConfig:
    return StationConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def load_thresholds(path: str | Path) -> Thresholds:
    return Thresholds.model_validate(yaml.safe_load(Path(path).read_text()))
