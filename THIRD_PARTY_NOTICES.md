# Third-party notices

## Model weights (not committed; fetched by `scripts/fetch_models.py`, pinned in `models/manifest.json`)

| Component | Source | Licence | Use |
|---|---|---|---|
| Grounding DINO tiny | IDEA-Research/grounding-dino-tiny | Apache-2.0 | pretrained proposals |
| SAM 2.1 hiera tiny | facebook/sam2.1-hiera-tiny | Apache-2.0 | masks |
| DINOv2 small | facebook/dinov2-small | Apache-2.0 | gallery embeddings |
| RT-DETR R18vd | PekingU/rtdetr_r18vd | Apache-2.0 | station detector candidate (COCO init) |

Verify the model cards for each pinned revision before commercial release (PDF p.11).

## Libraries
PyTorch (BSD-3), Transformers (Apache-2.0), ONNX / ONNX Runtime (Apache-2.0 / MIT), OpenCV (Apache-2.0),
ZXing-C++ (Apache-2.0), FastAPI (MIT), Pydantic (MIT), SciPy/NumPy (BSD-3), cryptography (Apache-2.0/BSD),
prometheus-client (Apache-2.0). Exact versions: `requirements/*.lock`.

## Photographs used for real-image evidence (not committed; `scripts/fetch_real_examples.py`)
- Open Food Facts contributors — product 7622202318078 image 1, CC BY-SA 3.0.
- "Milka Alpine Milk Chocolate chunks and wrapper" by Ubcule, Wikimedia Commons, CC BY-SA 4.0.

Overlays derived from them under `evidence/real/` are adaptations distributed under the same licences with
this attribution. Brand names and trade dress belong to their owners; they appear only as observed in these
photographs and are not used in any synthetic fixture.
