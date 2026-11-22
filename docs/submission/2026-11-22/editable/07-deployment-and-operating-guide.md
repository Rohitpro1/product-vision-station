# Deployment and operating guide

The public web path supports still images. The local installation adds a camera, replay, an offline event outbox and an optional central registry.

## Public smoke check

```bash
BASE=https://rohitpro1-pack-station-api.hf.space
curl -fsS "$BASE/readyz"
curl -fsS "$BASE/v1/meta"
curl -fsS -F file=@photo.jpg "$BASE/v1/detect"
```

Expect schema 1.0 and model bundle hosted-onnx-demo. /v1/meta reports CPUExecutionProvider, fixture-r1, thr-dev-0.5, up to 3 instances, 15 MiB upload, 40 MP decoded, 60 s deadline and concurrency 1. Frontend resizes JPEG to ≤1600 px; platform request limits may apply first. Warm requests of ~3.3 s (HF) and ~2.3 s (Vercel) are historical observations.

## Local reproducibility

```bash
# From a checkout of b56c2a7
bash scripts/bootstrap.sh --no-models
. .venv/bin/activate
pvs doctor
pvs fixtures all
pvs detect data/fixtures/scenes/fx-001000.png --ground-truth \
  data/fixtures/scenes/ground_truth.json
```

For pretrained vision, use bootstrap without --no-models, activate the environment, run python scripts/fetch_models.py and select configs/station.pretrained.yaml with --profile pretrained. The trained ONNX station profile additionally requires provisioned detector weights and a compatible catalogue. Verify the actual provider, bundle and thresholds before interpreting results.

| Condition | Operator response |
| --- | --- |
| 503 / failed readiness | Inspect the service logs, configured source SHA, archive digest and writable cache. Retry when ready, or select the verified fallback. An empty detections array has a different meaning. |
| Blur, glare, missing arrow | Use diffuse lighting, focus lock and a visible wrapper end. Keep reference artwork up to date. Commission near/far pixel scale with the installed lens. |
| Disconnected station | Pre-provision weights, capture calibration and a compatible bundle. SQLite WAL supports ordered retry and deduplication. Review backlog, frame age, drop counters and stale camera state. |
| Release regression | For hosted deployments restore the prior adapter/config and matching bundle URL + digest. For a station verify trust keys and use bundle rollback. Check /readyz and a frozen replay afterward. |

Sources: [Ops](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/OPERATOR_GUIDE.md), [API](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/API.md), [Checks](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/VERIFICATION.md), [HF meta](https://rohitpro1-pack-station-api.hf.space/v1/meta), [Vercel meta](https://pack-station-api.vercel.app/v1/meta)
