# Operator guide

## 1. Install (station or developer machine)

```bash
git clone <repo> && cd product-vision-station
./scripts/bootstrap.sh            # venv, pinned runtime+models+dev locks, editable install, fixtures, tests
python scripts/fetch_models.py    # pinned Apache-2.0 weights from Hugging Face + SHA-256 verification
pvs doctor                        # Python, torch/CUDA, ONNX Runtime providers, /dev/video*
```

GPU stations: install `onnxruntime-gpu` matching the CUDA driver, set `models.provider:
CUDAExecutionProvider` (or `TensorrtExecutionProvider`) and confirm with `pvs doctor` and the
`backend.detector` field, which names the provider that actually ran.

## 2. Run

| Task | Command |
|---|---|
| deterministic demo | `pvs fixtures all && pvs detect data/fixtures/scenes/fx-001000.png --ground-truth data/fixtures/scenes/ground_truth.json` |
| real pretrained still | `pvs --profile pretrained --config configs/station.pretrained.yaml detect photo.jpg --out out/real` |
| replay a clip | `pvs --profile pretrained --config configs/station.pretrained.yaml replay clip.mp4 --overlay --sample-hz 15` |
| live camera (CLI) | `pvs --profile pretrained --config configs/station.pretrained.yaml camera --device 0 --show` |
| service + preview | `pvs --profile pretrained --config configs/station.pretrained.yaml serve --device 0` → open http://127.0.0.1:8000/ |
| service on a replay | `pvs serve --video data/fixtures/video/crossing.mp4 --ground-truth data/fixtures/video/crossing_truth.json` |

The preview shows boxes, track IDs, identity level (green sku, amber family, grey unknown, red conflict,
orange ambiguous), face, a white **line** for an axis and a magenta **arrow** only for a supported heading,
the active profile/detector/bundle and **CAMERA STALE** when frames stop.

## 3. Bundles: install, verify, roll back

```bash
pvs bundle keygen --key ~/.config/pvs/release.key --pub deploy/trust/release.pub   # operator, once; key stays off-repo
pvs bundle build --src staging/bundle-src --out dist/bundle-2026-11-21 --bundle-id st01-2026-11-21 --key ~/.config/pvs/release.key \
     --meta '{"catalogue":"station-r3","thresholds":"thr-...","calibration":"cal-..."}'
pvs bundle verify  --src dist/bundle-2026-11-21 --trust deploy/trust
pvs --station-dir var/station bundle install --src dist/bundle-2026-11-21 --trust deploy/trust
pvs --station-dir var/station bundle rollback --trust deploy/trust
```

Install verifies signature, every file hash, unlisted files and schema compatibility *before* the atomic
`current` symlink swap; rollback re-verifies the previous bundle. Shadow a candidate first:
`scripts/shadow_compare.py out/active/results.jsonl out/candidate/results.jsonl` on the same replay.

## 4. Enrol a product (generic; no detector code changes)

1. Copy `catalogues/templates/product.example.json`, fill family, SKU, quantity/variant, artwork revision,
   provenance, usage rights; **leave `gtins` empty until verified against GS1/owner master data**.
2. Photograph 8–16 clean references across front/back; record A/B per `docs/reference_drawing.md` and
   variant-cue regions.
3. `pvs catalogue enrol --catalogue catalogues/station/catalogue.json --product my_product.json`
   (validation rejects duplicate/invalid GTIN mappings and missing references; failure leaves the catalogue untouched).
4. `pvs catalogue collisions --catalogue ... --embedder dinov2 --threshold 0.9` against all existing SKUs.
5. Replay the held-out sample, then `pvs catalogue rollback` if anything regresses.

## 5. Capture checklist (p.5, p.10)

- [ ] Camera locked: resolution, focus, exposure, white balance; log lens, mounting height, field of view and working distance in the station YAML `calibration`.
- [ ] Smallest pack long-axis ≥ 160 px at the far working distance; ≤ 40% occlusion for in-envelope presentations.
- [ ] Diffuse lighting; check foil highlights (consider polariser trial).
- [ ] Sessions: ≥ 3 backgrounds, several lighting/handling sessions, 12–20 physical positives across lots, 10–20 look-alike SKUs.
- [ ] Cover front, back, edge, upside-down, rapid turns, glare, folds, hand occlusion, identical bars together, non-targets, empty scenes.
- [ ] Assign pack/clip/session IDs *before* extracting frames; groups go to one split only.
- [ ] Hide turntable/fixture marks from model inputs.
- [ ] Run the four physical scenes: full rotation (unseen pack), front-to-back turnover in hand, crossing with a similar gold wrapper, removal then non-target.
- [ ] Re-run the reference-pack check after moving the camera, changing a lens/lighting or updating the catalogue.

## 6. Observability

Prometheus: `pvs_requests_total{status}`, `pvs_stage_ms{stage}`, `pvs_identity_total{level}`,
`pvs_live_scheduled_total`, `pvs_live_processed_total`, `pvs_live_dropped_frames`, `pvs_frame_age_ms`,
`pvs_model_ready`. Useful queries:

```promql
rate(pvs_live_processed_total[5m]) / rate(pvs_live_scheduled_total[5m])            # scheduled-sample coverage (gate ≥ 0.95)
histogram_quantile(0.95, sum by (le) (rate(pvs_frame_age_ms_bucket[5m])))          # capture→result p95 (gate ≤ 150 ms)
sum by (level) (rate(pvs_identity_total[15m]))                                     # unknown / conflict rates
histogram_quantile(0.95, sum by (le, stage) (rate(pvs_stage_ms_bucket[5m])))       # per-stage p95
```

Logs are structured (`logging` with module names) and contain metadata only; raw frames are never logged.
Retention default is metadata-only. Opted-in crops must have an owner, purpose, expiry and access control.

## 7. Failure behaviour

| Situation | Behaviour |
|---|---|
| model fails to load | `/readyz` 503, every result `model_failed`, preview banner shows status |
| camera disconnect / resolution change | new `stream_epoch`, tracks and evidence reset, preview **CAMERA STALE** |
| queue overload | latest-frame queue keeps newest; stale frames counted in `pvs_live_dropped_frames` |
| central DB down | inference and outbox continue; events delivered later in order, deduplicated by `event_id` |
| outbox full | delivered rows pruned; then refuse + high-water alarm + `data_loss_events` counter |
| corrupt bundle | install/verify refuses; active bundle unchanged; systemd `ExecStartPre` verify blocks start |
