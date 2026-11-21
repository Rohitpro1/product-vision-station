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

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$SUDO apt-get update -qq && $SUDO apt-get install -y -qq git unzip curl libgl1 libglib2.0-0 >/dev/null
[ -d "$REPO" ] || git clone --depth 1 https://github.com/Rohitpro1/product-vision-station "$REPO"
cd "$REPO"
# Ubuntu 22.04 ships Python 3.10; the pinned locks need >= 3.11 -> uv-managed Python 3.13
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -q -p 3.13 .venv-gpu
. .venv-gpu/bin/activate
# V100 = sm_70: newest PyTorch CUDA builds dropped Volta, so pin torch 2.6.0 built for CUDA 12.4 (includes sm_70)
uv pip install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
# torch 2.6 requires sympy==1.13.1: drop torch/torchvision/sympy pins from both locks
grep -vE '^(torch|torchvision|sympy)==' requirements/models.lock > /tmp/models-nottorch.lock
grep -vE '^sympy==' requirements/runtime.lock > /tmp/runtime-nosympy.lock
uv pip install -q -r /tmp/runtime-nosympy.lock -r /tmp/models-nottorch.lock sympy==1.13.1
uv pip install -q --no-deps -e .
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("gpu", torch.cuda.get_device_name(0), "arch list", torch.cuda.get_arch_list())
x = torch.randn(512, 512, device="cuda"); print("matmul ok", float((x @ x).sum()) == float((x @ x).sum()))
PY
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
