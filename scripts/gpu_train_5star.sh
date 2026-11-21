#!/usr/bin/env bash
# One-shot GPU run for the real-data 5 Star pipeline (Kaggle T4/P100, CyFuture V100, any CUDA 12.x Linux host).
#
# Prerequisite: the Roboflow COCO exports are unpacked under $DATA_ROOT (any folder names), e.g.
#   $DATA_ROOT/sct/{train,valid,test}/_annotations.coco.json  and  $DATA_ROOT/scancart/{...}
# Usage:  REPO=... OUT=... DATA_ROOT=... bash scripts/gpu_train_5star.sh [EPOCHS]     (default 40)
# Output: $OUT (train_report.json, eval_report.json, rtdetr_r18_package.onnx, gallery_5star.npz, logs)
set -euo pipefail
EPOCHS=${1:-40}
REPO=${REPO:-$HOME/product-vision-station}
OUT=${OUT:-$HOME/pvs-run}
DATA_ROOT=${DATA_ROOT:-$HOME/data}

# Host notebooks (e.g. Kaggle) inject PYTHONPATH/sitecustomize and user site-packages: isolate from them.
unset PYTHONPATH PYTHONSTARTUP PYTHONHOME PIP_TARGET PIP_PREFIX VIRTUAL_ENV CONDA_PREFIX || true
export PYTHONNOUSERSITE=1

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$SUDO apt-get update -qq >/dev/null 2>&1 || true
$SUDO apt-get install -y -qq git unzip curl libgl1 libglib2.0-0 >/dev/null 2>&1 || true
[ -d "$REPO" ] || git clone --depth 1 https://github.com/Rohitpro1/product-vision-station "$REPO"
cd "$REPO"

# The pinned locks need Python >= 3.11 (Ubuntu 22.04 ships 3.10) -> uv-managed Python 3.13.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
VENV="$REPO/.venv-gpu"
uv venv -q --python 3.13 --seed "$VENV"
PY="$VENV/bin/python"
UVPIP=(uv pip install -q --python "$PY")

# Pascal/Volta/Turing support: torch 2.6.0 built for CUDA 12.4 includes sm_60/sm_70/sm_75 kernels.
"${UVPIP[@]}" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
# torch 2.6 requires sympy==1.13.1: drop torch/torchvision/sympy pins from both locks
grep -vE '^(torch|torchvision|sympy)==' requirements/models.lock > /tmp/models-nottorch.lock
grep -vE '^sympy==' requirements/runtime.lock > /tmp/runtime-nosympy.lock
"${UVPIP[@]}" -r /tmp/runtime-nosympy.lock -r /tmp/models-nottorch.lock sympy==1.13.1
"${UVPIP[@]}" --no-deps -e .
"$PY" - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda build", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
print("gpu", torch.cuda.get_device_name(0), "arch list", torch.cuda.get_arch_list())
x = torch.randn(512, 512, device="cuda"); y = x @ x; torch.cuda.synchronize(); print("matmul ok", tuple(y.shape))
PY
"$PY" scripts/fetch_models.py --only rtdetr_r18 dinov2

mapfile -t EXPORTS < <(find "$DATA_ROOT" -name _annotations.coco.json -path '*/train/*' -printf '%h\n' | xargs -n1 dirname | sort -u)
echo "exports: ${EXPORTS[*]}"
[ "${#EXPORTS[@]}" -gt 0 ] || { echo "no COCO exports found under $DATA_ROOT"; exit 2; }
mkdir -p "$OUT"
"$PY" -m pvs.training.real5star prepare --exports "${EXPORTS[@]}" --out "$OUT/data" | tee "$OUT/prepare.log"
"$PY" -m pvs.training.real5star train --data "$OUT/data" --run "$OUT" --device cuda --epochs "$EPOCHS" \
    --batch 16 --size 640 --lr 2e-4 2>&1 | grep -v -i warning | tee "$OUT/train.log"
"$PY" -m pvs.training.real5star evaluate --data "$OUT/data" --run "$OUT" --device cuda --size 640 \
    --refs-per-class 40 2>&1 | grep -v -i warning | tee "$OUT/eval.log"
cp "$OUT/data/stats.json" "$OUT/data_stats.json"
tar -czf "$OUT/../pvs-run-artifacts.tgz" -C "$OUT" train_report.json eval_report.json data_stats.json \
    rtdetr_r18_package.onnx rtdetr_r18_package.json gallery_5star.npz prepare.log train.log eval.log
sha256sum "$OUT/../pvs-run-artifacts.tgz"
echo "DONE"
