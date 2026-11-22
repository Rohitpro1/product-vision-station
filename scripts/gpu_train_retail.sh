#!/usr/bin/env bash
# Retail detector + chocolate-brand gallery, end to end, on any CUDA 12 Linux host (built for Kaggle 2 x T4).
#   ROBOFLOW_API_KEY=... REPO=... WORK=... OUT=... bash scripts/gpu_train_retail.sh [EPOCHS]   (default 20)
#   SMOKE=1  -> full download + prepare, then 1 epoch x 30 steps and small evals, to prove the chain works
# Optional: NGPU, BATCH_PER_GPU (16), WORKERS (4), E2E_WORKERS (4)
# Output: $OUT/retail-v2-artifacts.tgz (reports + logs) and $OUT/bundle/ (models + config for the hosted API)
set -euo pipefail
EPOCHS=${1:-20}
REPO=${REPO:-$HOME/product-vision-station}
WORK=${WORK:-/tmp/pvs-retail}          # large scratch (datasets); not kept
OUT=${OUT:-$HOME/pvs-retail-out}       # small outputs to keep
BATCH_PER_GPU=${BATCH_PER_GPU:-16}
WORKERS=${WORKERS:-4}
E2E_WORKERS=${E2E_WORKERS:-4}
SMOKE=${SMOKE:-0}
[ -n "${ROBOFLOW_API_KEY:-}" ] || { echo "ROBOFLOW_API_KEY is not set"; exit 2; }

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
$SUDO apt-get update -qq >/dev/null 2>&1 || true
$SUDO apt-get install -y -qq git unzip curl libgl1 libglib2.0-0 >/dev/null 2>&1 || true
[ -d "$REPO" ] || git clone --depth 1 https://github.com/Rohitpro1/product-vision-station "$REPO"
cd "$REPO"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
UV_BIN=$(dirname "$(PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" command -v uv)")

# clean environment for every uv/python call (Kaggle injects PYTHONPATH + a sitecustomize that imports wrapt)
VENV="$REPO/.venv-gpu"
CLEAN=(env -i HOME="$HOME" LANG=C.UTF-8 TERM=dumb PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
       PATH="$VENV/bin:$UV_BIN:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
       LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" TMPDIR="${TMPDIR:-/tmp}" ROBOFLOW_API_KEY="$ROBOFLOW_API_KEY"
       HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4)
PY=("${CLEAN[@]}" "$VENV/bin/python" -s)
UVPIP=("${CLEAN[@]}" uv pip install -q --python "$VENV/bin/python")
"${CLEAN[@]}" uv venv -q --python 3.13 --seed "$VENV"
"${UVPIP[@]}" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
grep -vE '^(torch|torchvision|sympy)==' requirements/models.lock > /tmp/models-nottorch.lock
grep -vE '^sympy==' requirements/runtime.lock > /tmp/runtime-nosympy.lock
"${UVPIP[@]}" -r /tmp/runtime-nosympy.lock -r /tmp/models-nottorch.lock sympy==1.13.1 wrapt roboflow onnx
"${UVPIP[@]}" --no-deps -e .
"${PY[@]}" -c "import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__);assert torch.cuda.is_available();[print('gpu',i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
"${PY[@]}" scripts/fetch_models.py --only rtdetr_r18 dinov2

mkdir -p "$WORK" "$OUT/logs"
L="$OUT/logs"
echo "== download"; "${PY[@]}" -m pvs.training.retail download --out "$WORK/rf" 2>&1 | tee "$L/download.log" | tail -5
echo "== prepare";  "${PY[@]}" -m pvs.training.retail prepare --exports "$WORK/rf" --out "$WORK/data" 2>&1 | tee "$L/prepare.log"

BATCH=$((BATCH_PER_GPU * NGPU))
LR=$("${PY[@]}" -c "print(round(2e-4 * ($BATCH / 16) ** 0.5, 6))")
EXTRA=(); EVAL_LIMIT=2000; E2E_B=300; E2E_O=200
RUN="$WORK/run"
if [ "$SMOKE" = "1" ]; then EPOCHS=1; EXTRA=(--max-steps 30); EVAL_LIMIT=50; E2E_B=12; E2E_O=8; RUN="$WORK/run-smoke"; rm -rf "$RUN"; fi
if [ "$NGPU" -gt 1 ]; then LAUNCH=("${PY[@]}" -m torch.distributed.run --standalone --nproc_per_node="$NGPU"); else LAUNCH=("${PY[@]}"); fi
echo "== train detector: $NGPU GPU(s), batch $BATCH, lr $LR, $EPOCHS epochs"
"${LAUNCH[@]}" -m pvs.training.real5star train --data "$WORK/data" --run "$RUN" --device cuda --epochs "$EPOCHS" \
    --batch "$BATCH" --size 640 --lr "$LR" --workers "$WORKERS" --val-limit 400 "${EXTRA[@]}" 2>&1 \
    | grep -v -i -E 'warning|mismatch|detach' | tee "$L/train.log"
[ -f "$RUN/train_report.json" ] || { echo "training failed"; exit 4; }

echo "== export detector + test metrics"
"${PY[@]}" -m pvs.training.retail export --data "$WORK/data" --run "$RUN" --out "$OUT/detector" \
    --test-limit "$EVAL_LIMIT" 2>&1 | grep -v -i warning | tee "$L/export.log"
echo "== brand gallery"
"${PY[@]}" -m pvs.training.brands build --data "$WORK/data" --out "$OUT/brands" --device cuda 2>&1 \
    | grep -v -i -E 'warning|Loading weights' | tee "$L/brands.log"

echo "== bundle"
B="$OUT/bundle"; rm -rf "$B"; mkdir -p "$B/configs" "$B/models/trained" "$B/catalogues/brands"
cp configs/station.hosted-brands.yaml "$B/configs/station.hosted.yaml"
cp "$OUT/brands/thresholds_brands.yaml" "$B/configs/"
cp "$OUT/brands/catalogue_brands.json" "$B/catalogues/brands/catalogue.json"
cp "$OUT/detector/rtdetr_r18_package.onnx" "$OUT/brands/dinov2_small.onnx" "$OUT/brands/gallery_brands.npz" "$B/models/trained/"
cp models/manifest.json "$B/models/"
(cd "$B" && find . -type f | sort | xargs sha256sum) > "$OUT/bundle_sha256.txt"

echo "== end-to-end on held-out test photos"
"${PY[@]}" -m pvs.training.brands e2e --data "$WORK/data" --bundle "$B" --out "$OUT/brands" --workers "$E2E_WORKERS" \
    --max-branded "$E2E_B" --max-other "$E2E_O" 2>&1 | grep -v -i warning | tee "$L/e2e.log"

cp "$WORK/data/stats.json" "$OUT/data_stats.json"; cp "$WORK/rf/download_report.json" "$OUT/"
cp "$RUN/train_report.json" "$OUT/detector/"
tar -czf "$OUT/retail-v2-artifacts.tgz" -C "$OUT" data_stats.json download_report.json bundle_sha256.txt logs \
    detector/detector_report.json detector/train_report.json detector/rtdetr_r18_package.json \
    brands/brands_report.json brands/e2e_report.json brands/thresholds_brands.yaml brands/catalogue_brands.json
sha256sum "$OUT/retail-v2-artifacts.tgz"
du -sh "$B"/models/trained/*
echo "DONE"
