# Implementation and interfaces

The source repository contains the station package, the static frontend and deployment adapters. Only the execution profile selected by the service is active on a given request.

| Source area | Responsibility and boundary |
| --- | --- |
| capture / transforms / tracker | File EXIF, replay timestamps, latest-frame camera queue, epoch rollover, source-pixel mapping and stream association. Reconnect starts a new epoch. |
| detectors / pipeline / geometry / orientation | Grounding DINO, fixture or ONNX detector. Pipeline caps boxes, computes masks and checks crop quality. Mask shape supports axis. SIFT artwork matches supply direction, visible face and variant cues. |
| gallery / barcode / fusion | DINOv2, ONNX DINOv2 or colour histograms per profile. ZXing decodes and associates GTINs. Fusion owns SKU/family/ambiguous/conflict/unknown and expiry. |
| schema / api / render / cli | Pydantic FrameResult, FastAPI health/still/stream/metrics, preview and PNG overlays. Source, frame ID, coordinate frame, revision and evidence age travel with results. |
| catalogue / bundle / outbox | Reference enrolment and collisions; Ed25519 verification and atomic local release rollback; SQLite state transitions. PostgreSQL migrations support an optional central registry. |
| training/ + scripts/ | real5star handles v1 and shared DDP trainer; retail handles v2 manifests, dedup and export; brands and brand_map handle crop/full-frame evaluation. hub and progress publish training evidence. |
| frontend/ + deploy/ | Canvas overlays and backend selection; separate HF and Vercel adapters, Docker/Compose, systemd, Caddy and optional Postgres. render.yaml is a blueprint; Render deployment was deferred. |

## Response contract

POST /v1/detect accepts multipart file. Results use top-left-origin source pixel coordinates. Per instance: bbox_xyxy_px, mask_source, identity.level / rule / evidence, pose.axis and pose.heading (available, value_deg, period_deg), face and field provenance. JPEG, PNG, WebP and BMP are accepted by the application. Empty scenes return no_packages. Low-quality frames, malformed media, oversized requests and failed inference have explicit status paths.

Application-level routes include GET /healthz, /readyz, /v1/schema, /v1/meta and /metrics. Local stream mode adds /v1/live/latest, /v1/live/latest.jpg and WebSocket /v1/live. Deployments can expose different subsets. Use the included generated schema for exact field types and nullability.

Sources: [Pipeline](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/src/pvs/pipeline.py), [Fusion](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/src/pvs/fusion.py), [API](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/API.md), [Main](https://github.com/Rohitpro1/product-vision-station/tree/b56c2a781939fc1524a5b9d2e1ba297992d58bfb), [Ops](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/OPERATOR_GUIDE.md)
