---
title: Pack Station API
emoji: 📦
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Package detector API (RT-DETR-R18, ONNX) for Pack Station
---

# Pack Station API

This Space runs the backend for https://pack-station-psi.vercel.app. Source: https://github.com/Rohitpro1/product-vision-station (`deploy/hf-space`).

- **Detector:** RT-DETR-R18 fine-tuned for a single class, "package", on public retail photos on Kaggle (2 × T4).
- **Test results:** recall 0.964, precision 0.967.
- **Not validated:** Cadbury 5 Star identity and A→B heading. See `evidence/training/real5star_kaggle_t4x2` in the repo.

Endpoints: `GET /readyz`, `POST /v1/detect` (multipart `file`).
