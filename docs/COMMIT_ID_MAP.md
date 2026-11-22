# Commit ID map (history rewrite, 22 Nov 2026)

On 22 Nov 2026 the repository history was rewritten so that every commit is authored by the repository owner,
and generated trailer lines were removed from commit messages. File contents were not changed, apart from build-machine
absolute paths in ten evidence JSON files, which are now relative. The final tree is byte-identical to the one before the
rewrite.

Every commit therefore has a new ID. Documents written before the rewrite, including the frozen submission pack in
[`submission/2026-11-22/`](submission/2026-11-22/), quote the **old** IDs. That pack is checksummed and is not edited;
use this table to translate its IDs. Everything else in the repository now quotes the new IDs.

The `model-assets` branch was rewritten in the same way; the bundle file `pvs-hosted-bundle.tgz` is unchanged
(same blob, same SHA-256).

| Old ID | New ID | Commit |
|---|---|---|
| `07fc11a` | `b4fc787` | Frontend: Hugging Face Space as primary backend, Vercel as automatic fallback |
| `0a81971` | `63654dd` | Evidence: real-data GPU run on Kaggle 2xT4 (detector recall 0.964; 5 Star identity fails on public data) |
| `18c057b` | `4d532c9` | Hugging Face Space (Docker) for the API: same pinned package + model bundle as Vercel |
| `229b5a5` | `85cae51` | Retail runner: separate smoke checkpoint dir; default 20 epochs |
| `28b68ce` | `b3d96ba` | Evidence and documentation for implementation revisions 6c36061 / ddadf2e |
| `3bdf374` | `13c737f` | GPU runner for the real-data 5 Star training/evaluation |
| `3fb7c75` | `8ac8914` | HF Space: leave web/pydantic stack to Gradio's resolver (gradio[mcp] needs pydantic<=2.12.5) |
| `43b8f0e` | `1e3bfc0` | Retail runner: fix RUN default |
| `473b608` | `6c36061` | Product vision station: fixture/pretrained/station pipeline, API, outbox, bundles, training and evaluation |
| `48ca424` | `c41f621` | Docs: Hugging Face Space as primary API with Vercel fallback (D21, live checks) |
| `52f821a` | `e7c31aa` | GPU runner: isolate from host PYTHONPATH/sitecustomize, call the venv interpreter explicitly |
| `5bac355` | `0d253b1` | Retail v2 training: multi-dataset detector + chocolate-brand DINOv2 gallery |
| `5edc21f` | `f39e27a` | Docs: live hosted demo (Vercel site + API), decisions D17-D20, hosted check |
| `5ffae01` | `f09d395` | HF Space: launch via Gradio, attach REST API + CORS in App.create_app (port 7860 is owned by the Space runtime) |
| `6b0b828` | `254455f` | GPU runner: drop sympy pin from the lock filter (torch 2.6 needs sympy 1.13.1) |
| `7453926` | `817ad2f` | Hosted API: trained ONNX detector profile, fast ROI GrabCut, Vercel app, client-side downscale |
| `84aaa68` | `ddadf2e` | Tracker: never re-associate a track after a detection gap; reappearing objects get a new ID |
| `85351e4` | `55edb10` | Verification: record GitHub publication, CI run and fresh-clone check |
| `8f01385` | `c71dda5` | docs: add engineering submission pack (22 Nov 2026) and update READMEs |
| `ad0819a` | `e87713c` | Retail runner: live progress + ETA, live HF progress page, auto-upload of results |
| `b143625` | `461ff7d` | Runtime bundle for the hosted API (model v1) |
| `b184c0f` | `8caeeef` | HF Space: pin package to 0118de2 (includes preview.html) |
| `b26fb8b` | `87d50c7` | Docs: correct Space source commits |
| `b56c2a7` | `e776ab8` | docs: complete project handoff (requests, live system, training runs, datasets, architecture diagrams, scripts, hosting, fixes, open items) |
| `b5ee790` | `9dbc97a` | GPU runner: train on all GPUs with DDP (torchrun), threaded data loading, clean env for Kaggle |
| `cb72a6e` | `65985b5` | Frontend: load example assets from the public repo; configurable examplesBase |
| `cf3992b` | `0118de2` | Package preview.html; HF Space switched to Gradio SDK (free tier) with the API mounted |
| `dfc0dea` | `b06f7a4` | GPU runner: also drop sympy pin from runtime.lock (resolution verified) |
| `e4dd5f8` | `d41493f` | GPU runner: uv Python 3.13, torch 2.6.0+cu124 for V100 (sm_70), apt deps, CUDA self-check |
| `ec36a0c` | `641c3f2` | Frontend defaults to the hosted API; Vercel API pinned to 817ad2f + bundle sha256 |
| `ed64b5e` | `6d864f6` | Retail runner: roboflow in its own venv (typer conflict made uv pick roboflow 0.1.1 without requests) |
| `f72938a` | `57a4e1a` | Real-data 5 Star training/eval pipeline, static frontend, backend CORS, Render blueprint |
| `ff3e3b8` | `76732d1` | Verification: record clean-clone results and publication status |
