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
short_description: Package detector + brand API (ONNX) for Pack Station
---

# Pack Station API

This Space is the backend for https://pack-station-psi.vercel.app. Source: https://github.com/Rohitpro1/product-vision-station (`deploy/hf-space`).

The model served is the release named in `release.json` on the repository's `model-assets` branch. The Space checks
it every 10 minutes and switches over without a restart (see `app.py`).

- **Detector (release v2):** RT-DETR-R18 fine-tuned for one class, "package", on 36 public retail projects (Kaggle, 2 × T4).
  Test recall 0.808, precision 0.801 on 2,000 dense shelf photos.
- **Brands:** DINOv2-small gallery naming Cadbury Dairy Milk, Cadbury Gems, KitKat, Bounty and Kinder when confident.
- **Not validated:** Cadbury 5 Star identity and A→B heading. See `evidence/training/retail_v2_kaggle_t4x2` in the repo.

Endpoints: `GET /readyz`, `POST /v1/detect` (multipart `file`). The page at `/` is a small upload demo.
