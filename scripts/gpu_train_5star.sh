#!/usr/bin/env bash
# One-shot GPU run for the real-data 5 Star pipeline (CyFuture V100 / any CUDA 12.x Ubuntu host).
#
# Prerequisite: the two Roboflow COCO exports are unpacked under ~/data/ (any folder names), e.g.
#   ~/data/sct/{train,valid,test}/_annotations.coco.json  and  ~/data/scancart/{...}
# Usage:  bash scripts/gpu_train_5star.sh [EPOCHS]        (default 40)
# Output: ~/pvs-run/  (train_report.json, eval_report.json, rtdetr_r18_package.onnx, gallery_5star.npz, logs)
set -euo pipefail
EPOCHS=${1:-40}
REPO=${REPO:-$HOME/product-vision-station}
OUT=${OUT:-$HOME/pvs-run}
DATA_ROOT=${DATA_ROOT:-$HOME/data}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
[ -d "$REPO" ] || git clone --depth 1 https://github.com/Rohitpro1/product-vision-station "$REPO"
cd "$REPO"
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install -q --upgrade pip
pip install -q --index-url https://download.pytorch.org/whl/cu124 torch torchvision
grep -vE '^(torch|torchvision)==' requirements/models.lock > /tmp/models-nottorch.lock
pip install -q -r requirements/runtime.lock -r /tmp/models-nottorch.lock
pip install -q --no-deps -e .
python -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/fetch_models.py --only rtdetr_r18 dinov2

mapfile -t EXPORTS < <(find "$DATA_ROOT" -name _annotations.coco.json -path '*/train/*' -printf '%h\n' | xargs -n1 dirname | sort -u)
echo "exports: ${EXPORTS[*]}"
mkdir -p "$OUT"
python -m pvs.training.real5star prepare --exports "${EXPORTS[@]}" --out "$OUT/data" | tee "$OUT/prepare.log"
python -m pvs.training.real5star train --data "$OUT/data" --run "$OUT" --device cuda --epochs "$EPOCHS" \
    --batch 16 --size 640 --lr 2e-4 2>&1 | grep -v -i warning | tee "$OUT/train.log"
python -m pvs.training.real5star evaluate --data "$OUT/data" --run "$OUT" --device cuda --size 640 \
    --refs-per-class 40 2>&1 | grep -v -i warning | tee "$OUT/eval.log"
cp "$OUT/data/stats.json" "$OUT/data_stats.json"
tar -czf "$OUT/../pvs-run-artifacts.tgz" -C "$OUT" train_report.json eval_report.json data_stats.json \
    rtdetr_r18_package.onnx rtdetr_r18_package.json gallery_5star.npz prepare.log train.log eval.log
sha256sum "$OUT/../pvs-run-artifacts.tgz"
echo "DONE - terminate the GPU instance now."
