# Release conditions and custody

Review disposition: submit the software delivery and the v1 detector evidence. A field-release decision requires the named evidence below. Owner approval and acceptance signatures have not been supplied.

| Responsible role | Remaining evidence or decision | Acceptance record |
| --- | --- | --- |
| Vision / data lead | Capture 12–20 physical target packs across lots, 10–20 look-alikes and independent pack/session splits. Enrol approved front/back templates, verified variant data and physical A/B labels. | Ground-truth manifest plus sealed field report. Include at least 299 independent clean negatives if demonstrating a ≤1% one-sided upper bound with zero false accepts. |
| Station / QA lead | Lock lens, exposure, white balance, distance and lighting. Trial ≥160 px pack length and ≤40% occlusion. Test full rotation, turnover, crossing, removal and reconnect. | Report axis/heading observability and coverage, p95 error, flip rate, joint yield, p99 latency, 8-hour memory and event integrity on named hardware. |
| Release owner | Review v2 per-project versions, licences, retained classes, detector and crop/full-image brand reports. Keep brand family claims distinct from verified exact SKUs. | Freeze candidate; replay both hosts and compare per-instance outputs. Require approval to promote and retain the known-good v1 hash for rollback. |
| Platform owner | Build/test the Docker image on a supported host. Verify storage, access and retention before accepting sensitive photos; reconcile differing gallery IDs. Check disposition of attempted CyFuture VMs. | Build record, cleanup/retention evidence and commissioning report. Kaggle and hosted tiers supply no demonstrated station SLA. |

## Data and licensing disclosure

GitHub source is publicly readable. LICENSE reserves rights to Rohit Kumar P, while the Space card lists Apache-2.0. Reconcile the metadata before external redistribution. Upstream model and dataset notices apply separately. The package includes measurements and manifests; raw training photos, tokens and model checkpoints are distributed separately where authorised.

The public APIs accept requests without a configured bearer token in the observed demo. CORS controls browser origins. The frontend can retain an operator-supplied token in localStorage. Direct Gradio uploads may create temporary files, including a resized upload.jpg. The deployed source therefore does not support a blanket zero-retention claim.

Sources: [Licence](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/LICENSE), [Notices](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/THIRD_PARTY_NOTICES.md), [Space](https://huggingface.co/spaces/Rohitpro1/pack-station-api/tree/465a53b02b56ad19a187164d9ebdb27d5c7f1e86), [Ops](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/OPERATOR_GUIDE.md), [Requirements](https://github.com/Rohitpro1/product-vision-station/blob/b56c2a781939fc1524a5b9d2e1ba297992d58bfb/docs/REQUIREMENTS_EVIDENCE.md)
