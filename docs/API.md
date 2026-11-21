# API

`pvs serve` starts FastAPI on **127.0.0.1:8000** by default (loopback; no auth). For a shared station set
`api_token_env` in the station YAML (clients send `Authorization: Bearer <token>`) and terminate TLS in
the proxy (`deploy/Caddyfile`). The generated JSON Schema is served at `GET /v1/schema`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/detect` | multipart `file` → `FrameResult` (one still) |
| GET | `/healthz` | liveness |
| GET | `/readyz` | 200 when models loaded; 503 `model_failed` with reason |
| GET | `/v1/meta` | profile, backends, bundle, catalogue revision, thresholds, limits, products |
| GET | `/v1/schema` | JSON Schema of `FrameResult` (schema 1.0) |
| GET | `/v1/live/latest`, `/v1/live/latest.jpg` | latest live/replay result + overlay |
| WS | `/v1/live` | push each new live `FrameResult` |
| GET | `/metrics` | Prometheus exposition |
| GET | `/` | local preview page |

## Limits and errors

Defaults (`LimitsCfg`): max upload 15 MiB, max decoded pixels 40 MP, formats jpeg/png/webp/bmp, at most 3
instances per frame (extra → `reason: instances_capped`), request deadline 60 s, concurrency 1 (bounded
semaphore; results past the deadline are discarded with 504).

| HTTP | `status` | `error` | Meaning |
|---|---|---|---|
| 200 | `ok` | — | inference ran; `instances` ≥ 1 |
| 200 | `no_packages` | — | inference ran; `instances: []` |
| 200 | `poor_quality` | — | `reason` quality_dark / quality_blurred / quality_saturated |
| 400 | `decode_failed` | `undecodable`, `empty_upload` | bytes are not an image |
| 413 | `decode_failed` | `payload_too_large`, `too_many_pixels` | limits |
| 415 | `decode_failed` | `unsupported_format` | e.g. TIFF |
| 503 | `model_failed` | — | backend unavailable; no predictions |
| 504 | `model_failed` | `deadline_exceeded` | work discarded |

Frame ownership: the upload buffer is decoded once into the declared source frame (EXIF applied); the
service keeps no frame after the response. Raw frames are never logged.

## Example

```bash
curl -s -F file=@evidence/examples/exact_sku_visual_overlay.jpg http://127.0.0.1:8000/v1/detect | jq .
```

Abbreviated real response (fixture profile, `evidence/examples/exact_sku_barcode_back_face.json`):

```json
{
  "schema_version": "1.0", "status": "ok", "profile": "fixture",
  "backend": {"detector": "scripted-fixture", "embedder": "colour-hist-v1", "template": "sift-homography-v1", "barcode": "zxing-cpp", "ocr": "unavailable"},
  "model_bundle": "dev-unbundled", "catalogue_revision": "fixture-r1", "thresholds_version": "thr-dev-0.5", "fusion_rules_version": "fusion-v1.5",
  "camera": "file", "stream_epoch": 0, "frame_id": 0, "image_size_px": [960, 540], "coordinate_frame": "image_rectified",
  "instances": [{
    "bbox_xyxy_px": [270.0, 210.0, 690.0, 330.0], "mask_source": "manual",
    "identity": {"level": "sku", "family": "fixture:zephyr", "sku": "fixture:zephyr-classic-40g", "gtin": "02000000000015",
                 "rule": "R2_verified_gtin", "score_type": "rule_based",
                 "evidence": [{"kind": "barcode", "value": "2000000000015", "polygon_px": [[...]]}, {"kind": "gallery"}, {"kind": "template"}]},
    "pose": {"frame": "image_rectified", "face": "back",
             "axis": {"value_deg": 0.0, "period_deg": 180, "available": true, "source": "mask:manual"},
             "heading": {"value_deg": 0.0, "period_deg": 360, "available": true, "source": "sift-homography-v1"},
             "endpoints_px": {"A": [270.0, 270.0], "B": [690.0, 270.0]}}
  }]
}
```

Every field carries its own `observed_at` / `age_ms`; a held value (`reason: "held"`, `rule: R6_track_hold`)
exposes its age. Gallery scores are `raw_cosine_uncalibrated` until a calibrated thresholds file sets
`calibrated: true`. When a field is unavailable its `value_deg` is `null` and `reason` is one of the codes in
`geometry.Reason` (missing: `endpoint_hidden`, `no_mask`, `evidence_expired`; ambiguous: `axis_unstable`,
`near_round`, `tied_candidates`, `conflicting_code`, `clustered_matches`, `unstable_temporal`, …).
