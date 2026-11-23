# Laptop station (YOLO26-seg + OCR)

One program runs on a laptop with 4 GB RAM and finds every chocolate pack in the camera frame. For each pack it gives:

- its exact outline (segmentation mask + polygon) and its angle
- its brand, from the YOLO class, confirmed by the printed name when OCR can read it
- the printed MRP, manufacturing / expiry / best-before dates and the batch number, when they can be read

The laptop needs no PyTorch and no Ultralytics: only ONNX Runtime, OpenCV and RapidOCR, about 34 MB of model
files and 350 MB of memory while running.

## How it works

| Step | Model | When it runs |
|---|---|---|
| Detect + segment + brand | YOLO26n-seg, our classes (`package` + brands), NMS-free ONNX | every frame (GPU if present, else CPU) |
| Track | box overlap across frames | every frame |
| OCR | PP-OCRv5 mobile detector + English recogniser (RapidOCR) | once per new pack, in a background thread, on the upright pack crop |
| Fields | `pvs.edge.fields`: MRP, dates, batch, brand name | after each OCR read |

The brand shown is the model's class when its score passes `brand_conf`. That threshold is chosen on the validation photos so that at most 1 % of names are wrong. Below it, the pack is shown as `package`.

Printed text can confirm the model's brand, or name a pack the model left as `package`. If text and model disagree, the pack is marked `check` and nothing is guessed. A date or MRP that does not parse cleanly is left out.

## Training (Kaggle, 2 × T4)

`scripts/gpu_train_yolo.sh` runs everything in one cell:

1. Data: the Roboflow projects in `configs/brand_datasets.yaml` plus Open Food Facts front-of-pack close-ups, deduplicated, with leakage-safe splits.
2. Masks: SAM2 turns each labelled box into an outline. Masks that don't fit their box fall back to the rectangle, and the report counts both.
3. A smoke run of the whole chain, then full training, time-boxed to the session.
4. Export to 640 and 480 px, in fp32 for CPU and fp16 for GPU.
5. Test results:
   - box and mask mAP from Ultralytics;
   - our own test with the exact laptop code: brand right / wrong, other packs called a brand, pack recall, mask agreement;
   - OCR brand agreement;
   - speed and memory on the T4 GPU and on 4 CPU threads under a hard 4 GB cap;
   - the live v2 site on the same test split, for comparison.

## On the laptop

```bash
git clone https://github.com/Rohitpro1/product-vision-station && cd product-vision-station
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements/edge.txt
pip install "onnxruntime-gpu[cuda,cudnn]==1.32.0"         # NVIDIA GPU;  CPU only: pip install onnxruntime==1.32.0
pip install opencv-python                                  # for the live window (replaces the headless build)
pip install --no-deps -e .
# get the bundle: edge_bundle.tgz from huggingface.co/Rohitpro1/pack-station-training (results/), then
tar -xzf edge_bundle.tgz                                    # -> bundle/

python -m pvs.edge.station run   --bundle bundle --source 0                     # webcam, press q to quit
python -m pvs.edge.station run   --bundle bundle --source shelf.jpg --out shelf_out.jpg --json shelf.json
python -m pvs.edge.station bench --bundle bundle --images some_photos/ --mem-limit-gb 4
```

`--device cpu` forces the CPU. `--no-ocr` skips text reading. `--conf` changes the detection threshold.

The GPU uses `model_480_fp16.onnx` and the CPU uses `model_480.onnx`; both are set in `bundle/edge.json`. If a pack is small in the frame, switch `model` to the 640 file for better accuracy at the cost of speed.

## Speed

The real numbers come from the Kaggle run (`yolo/bench_t4_gpu.json`, `yolo/bench_cpu_4gb.json`). Measure on the laptop itself with `bench`.

For reference, Ultralytics publishes 53 ms for YOLO26n-seg at 640 px on a 2 GHz server CPU with ONNX. Here, on 2 slow server cores at 480 px, our runtime took about 60 ms per frame, with a 343 MB memory peak under a hard 4 GB limit. A laptop NVIDIA GPU should bring detection + segmentation into single-digit to low-tens of milliseconds.

OCR takes roughly 0.1–0.4 s per pack on CPU. It runs once per pack in the background, so live video does not wait for it.

## Licences: read before shipping

- **Ultralytics YOLO26 (training code and the pretrained weights our model starts from) is AGPL-3.0.** To use the trained model inside a closed company product, the company either publishes that product's source under AGPL-3.0 or buys an Ultralytics Enterprise licence. This repository is public, so the project itself is fine. The decision for the employer's product is theirs.
- PP-OCRv5 / PaddleOCR and RapidOCR: Apache-2.0. SAM2 (only used to make training masks): Apache-2.0.
- Training images (Roboflow Universe projects, Open Food Facts photos CC BY-SA) are used for training and evaluation only and are not redistributed. The bundle contains model weights, not images.
