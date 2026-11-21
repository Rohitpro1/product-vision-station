# Requirements → evidence

Keyed to the architecture PDF pages. **State** legend: ✅ implemented and verified at the stated level ·
🟡 implemented, verified only on fixtures / smoke · ⛔ blocked on a named prerequisite.

| PDF | Requirement | Component | Test / experiment | Evidence path | State |
|---|---|---|---|---|---|
| p.1 | box + family + exact SKU + arrow for a front-facing pack | `pipeline.py`, `fusion.py` | `test_front_view_gives_sku_and_heading` | `evidence/examples/exact_sku_visual*` | ✅ fixture · ⛔ real 5 Star |
| p.1 | logo under thumb: keep box + axis; clear arrow when uncertain | `geometry.PeriodicSmoother`, `_pose` | `test_heading_180_swap_clears`, `test_evidence_expires_after_gap` | `evidence/examples/family_variant_cue_hidden*` | ✅ fixture |
| p.1 | two similar gold packs: one track each, barcode never certifies the neighbour | `tracker.py`, `barcode.associate` | `test_occluding_swap_drops_late_read_and_clears_evidence`, `test_replay_crossing_tracks_and_events` | `evidence/replay/crossing/*` | ✅ fixture |
| p.1 | unknown package → box + axis; empty scene → `[]` | `fusion R7`, `FrameStatus.NO_PACKAGES` | `test_distractor_unknown_with_axis_and_no_arrow`, `test_empty_frame_returns_empty_array` | `evidence/examples/unknown_*`, `empty_frame*`, `evidence/real/5star_unenrolled*` | ✅ fixture + real photo |
| p.2 | acquire: EXIF once, mirror policy, crop→source transforms | `capture.py`, `transforms.py` | `test_transforms.py` (8 EXIF orientations, letterbox, crop at edge, mirror) | tests | ✅ |
| p.2 | locate: generic package detector, original-res crops | `detectors/hf_models.py`, `detectors/rtdetr.py`, `training/detector.py` | `test_models.py` (opt-in), `pvs train detector-smoke` | `evidence/eval/pretrained_sealed20_seed12000`, `evidence/training/rtdetr_smoke.json` | 🟡 pretrained works; fine-tuned detector needs real data |
| p.2 | associate: ByteTrack-style, reinit after restart / gap | `tracker.py` | `test_tracker_fusion.py` | tests | ✅ |
| p.2 | measure: gallery + geometry per crop; barcode bound to capture time + instance | `gallery.py`, `orientation.py`, `barcode.py` | `test_back_barcode_resolves_sku_and_keeps_heading` | `evidence/examples/exact_sku_barcode_back_face*` | ✅ fixture |
| p.2 | publish: per-output thresholds, expiry, WebSocket, idempotent events | `api.py`, `outbox.py` | `test_outbox_*`, `test_api_*` | `evidence/api/*` | ✅ |
| p.3 | axis [0,180), heading [0,360), A→B, face separate, reason codes | `geometry.py`, `schema.py` | `test_geometry.py` (cardinals, 179/1, 359/1, swaps) | tests | ✅ |
| p.3 | world / metric 3D only as extensions | `docs/EXTENSIONS.md` | — | doc | ✅ (disabled by design) |
| p.4 | Grounding DINO + SAM 2 baseline; DINOv2 gallery; RT-DETR candidate; template direction | `models/manifest.json` (pinned + SHA-256) | `scripts/fetch_models.py` | manifest | ✅ pretrained inference runs |
| p.4 | first comparison: reference matching vs specialised detector on identical clips | `scripts/shadow_compare.py`, `pvs eval` | — | — | ⛔ needs real held-out clips |
| p.5 | grouped splits, leakage check, CVAT labels, reviewer protocol | `training/data.py`, `docs/TRAINING_EVALUATION.md` | `test_group_split_*`, `test_leakage_check_*`, `test_cvat_coco_export` | tests | ✅ tooling · ⛔ 0 real packs captured |
| p.5 | landmark/face head with visibility loss, label-consistent augmentation | `training/landmark.py` | `test_landmark_smoke_resume_and_onnx_parity` | `evidence/training/*` | 🟡 smoke only |
| p.6 | GTIN validation, one polygon → one instance, conflict review | `barcode.py`, `fusion.py` | `test_gtin_validation`, `test_association_*`, `test_code_for_other_product_*` | `evidence/examples/conflict_foreign_code*` | ✅ fixture |
| p.6 | result contract with schema, camera, epoch, frame, ages, bundle/catalogue | `schema.py` | `test_api_detect_validates_against_schema`, `test_cli_detect_and_decode_failure` | `docs/API.md` | ✅ |
| p.6 | catalogue: verified GTINs, revisions, references, collision tests | `catalogue.py` | `test_catalogue_*`, `test_enrol_and_rollback` | `catalogues/fixture/catalogue.json` | ✅ · real GTIN unset (unverified) |
| p.7 | ONNX Runtime serving, fallback provider reported | `detectors/rtdetr.py` | detector smoke parity | `evidence/training/rtdetr_smoke.json` | 🟡 CPU only; TensorRT ⛔ no NVIDIA GPU |
| p.7 | FastAPI + WebSocket; Postgres catalogue/releases; SQLite WAL outbox | `api.py`, `migrations/postgres`, `outbox.py` | `test_postgres_migrations_apply_and_check` (local PG 16) | `evidence/api/*` | ✅ |
| p.7 | signed bundle, shadow, rollback | `bundle.py` | `test_bundle_*` | tests | ✅ |
| p.7 | Docker + systemd watchdog; Prometheus/OTel; metadata-only retention | `deploy/*`, `api.py /metrics` | — | — | 🟡 written; Docker image not built (no daemon) |
| p.8 | acceptance gates with raw counts, CIs, insufficient-evidence | `evaluation.py` | `test_one_sided_bound_matches_design_note`, `test_insufficient_evidence_*` | `evidence/eval/*/report.md` | ✅ harness · ⛔ real presentations |
| p.8 | runtime: 15 fps, p95 ≤150 ms, 8-hour soak with reconnects | `api.LiveWorker`, metrics | runbook in `docs/VERIFICATION.md` | — | ⛔ no camera / GPU station |
| p.9 | quote basis kept separate from measurements | `docs/DECISIONS.md` | — | doc | ✅ |
| p.10 | commissioning: reference-pack check gates rollout | `docs/OPERATOR_GUIDE.md` §5 | — | — | ⛔ needs station |
| p.11 | record checkpoint, source commit, versions, notices | `models/manifest.json`, `THIRD_PARTY_NOTICES.md`, eval `run.env` | — | reports | ✅ |
