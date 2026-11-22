---
title: Pack Station API
emoji: 📦
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Package detector API (RT-DETR-R18, ONNX) for Pack Station
---

# Pack Station API

This Space is the backend for https://pack-station-psi.vercel.app. Source: https://github.com/Rohitpro1/product-vision-station (`deploy/hf-space`).

- **Detector:** RT-DETR-R18 fine-tuned for one class, "package", on public retail photos (Kaggle, 2 × T4).
- **Test results:** recall 0.964, precision 0.967.
- **Not validated:** Cadbury 5 Star identity and A→B heading. See `evidence/training/real5star_kaggle_t4x2` in the repo.

Endpoints: `GET /readyz`, `POST /v1/detect` (multipart `file`). The page at `/` is a small upload demo.
