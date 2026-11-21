"""Versioned response contract (architecture PDF p.6, build prompt section 7).

Coordinate frame ``image_rectified``: upright (EXIF applied once), unmirrored
source pixels; origin top-left, x right, y down, clockwise-positive angles.
Boxes are ``[x0, y0, x1, y1]`` in source pixels (float). Polygons are lists of
``[x, y]`` source pixels in clockwise order starting top-left where defined.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class FrameStatus(str, Enum):
    OK = "ok"  # inference ran; zero or more instances
    NO_PACKAGES = "no_packages"  # inference ran; nothing detected (instances == [])
    POOR_QUALITY = "poor_quality"  # frame too dark/blurred/saturated to trust
    MODEL_FAILED = "model_failed"  # a required model is unavailable; no predictions made
    DECODE_FAILED = "decode_failed"  # input could not be decoded


class IdentityLevel(str, Enum):
    SKU = "sku"
    FAMILY = "family"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    AMBIGUOUS = "ambiguous"


class ScoreType(str, Enum):
    RAW_COSINE = "raw_cosine_uncalibrated"
    CALIBRATED = "calibrated_probability"
    RULE = "rule_based"
    NONE = "none"


class Face(str, Enum):
    FRONT = "front"
    BACK = "back"
    EDGE = "edge"
    UNKNOWN = "unknown"


class Profile(str, Enum):
    FIXTURE = "fixture"
    PRETRAINED = "pretrained"
    STATION = "station"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["gallery", "barcode", "ocr", "template", "track_history", "scripted_fixture"]
    value: str | None = None
    score: float | None = None
    score_type: ScoreType = ScoreType.NONE
    frame_id: int | None = None
    observed_at: str | None = None
    age_ms: float | None = None
    polygon_px: list[list[float]] | None = None


class Candidate(BaseModel):
    sku: str
    family: str
    side: str
    score: float
    reference_revision: str


class Identity(BaseModel):
    level: IdentityLevel
    family: str | None = None
    sku: str | None = None
    gtin: str | None = None
    score: float | None = None
    score_type: ScoreType = ScoreType.NONE
    margin: float | None = None
    rule: str | None = None
    reason: str = "ok"
    evidence: list[Evidence] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    observed_at: str | None = None
    age_ms: float | None = None


class Field_(BaseModel):
    value_deg: float | None = None
    period_deg: Literal[180, 360]
    available: bool
    score: float | None = None
    reason: str = "ok"
    source: str | None = None
    observed_at: str | None = None
    age_ms: float | None = None


class Pose(BaseModel):
    frame: Literal["image_rectified"] = "image_rectified"
    face: Face = Face.UNKNOWN
    face_score: float | None = None
    face_reason: str = "ok"
    axis: Field_
    heading: Field_
    endpoints_px: dict[str, list[float]] | None = None  # {"A": [x,y], "B": [x,y]} when heading available


class Instance(BaseModel):
    instance_id: str
    track_id: str | None = None
    bbox_xyxy_px: list[float]
    det_score: float
    det_source: str
    mask_source: Literal["segmentation", "manual", "geometry_fallback_threshold", "geometry_fallback_box", "none"]
    identity: Identity
    pose: Pose


class Quality(BaseModel):
    status: Literal["ok", "dark", "saturated", "blurred"]
    mean_luma: float
    saturated_frac: float
    sharpness: float


class FrameResult(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: FrameStatus
    reason: str = "ok"
    profile: Profile
    backend: dict[str, str]
    model_bundle: str
    catalogue_revision: str
    thresholds_version: str
    fusion_rules_version: str
    camera: str
    stream_epoch: int
    frame_id: int
    captured_at: str
    processed_at: str
    source: str
    image_size_px: list[int]  # [width, height] of the declared source frame
    coordinate_frame: Literal["image_rectified"] = "image_rectified"
    quality: Quality | None = None
    timing_ms: dict[str, float] = Field(default_factory=dict)
    instances: list[Instance] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: FrameStatus
    error: str
    detail: str
