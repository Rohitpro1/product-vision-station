#!/usr/bin/env bash
# One-shot GPU run for the real-data 5 Star pipeline (Kaggle T4x2/P100, CyFuture V100, any CUDA 12.x Linux host).
# Uses every visible GPU: training runs under torchrun (DDP), one process per GPU.
#
# Prerequisite: the Roboflow COCO exports are unpacked under $DATA_ROOT (any folder names), e.g.
#   $DATA_ROOT/sct/{train,valid,test}/_annotations.coco.json  and  $DATA_ROOT/scancart/{...}
# Usage:  REPO=... OUT=... DATA_ROOT=... bash scripts/gpu_train_5star.sh [EPOCHS]     (default 40)
#   optional: NGPU=1 (force one GPU), BATCH_PER_GPU=16, WORKERS=4 (data workers per GPU)
# Output: $OUT (train_report.json, eval_report.json, rtdetr_r18_package.onnx, gallery_5star.npz, logs)
set -euo pipefail
EPOCHS=${1:-40}
REPO=${REPO:-$HOME/product-vision-station}
OUT=${OUT:-$HOME/pvs-run}
DATA_ROOT=${DATA_ROOT:-$HOME/data}
BATCH_PER_GPU=${BATCH_PER_GPU:-16}
WORKERS=${WORKERS:-4}

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
[ "$NGPU" -ge 1 ] || { echo "no GPU visible"; exit 3; }
$SUDO apt-get update -qq >/dev/null 2>&1 || true
$SUDO apt-get install -y -qq git unzip curl libgl1 libglib2.0-0 >/dev/null 2>&1 || true
[ -d "$REPO" ] || git clone --depth 1 https://github.com/Rohitpro1/product-vision-station "$REPO"
cd "$REPO"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
UV_BIN=$(dirname "$(PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" command -v uv)")

# Host notebooks (Kaggle) inject PYTHONPATH, a sitecustomize that imports wrapt, user site-packages, etc.
# Every uv/python call below runs from an empty environment with only what it needs.
VENV="$REPO/.venv-gpu"
CLEAN=(env -i HOME="$HOME" LANG=C.UTF-8 TERM=dumb PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
       PATH="$VENV/bin:$UV_BIN:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
       LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" TMPDIR="${TMPDIR:-/tmp}"
       HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
       NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}")
PY=("${CLEAN[@]}" "$VENV/bin/python" -s)
UVPIP=("${CLEAN[@]}" uv pip install -q --python "$VENV/bin/python")

# The pinned locks need Python >= 3.11 (Ubuntu 22.04 ships 3.10) -> uv-managed Python 3.13.
"${CLEAN[@]}" uv venv -q --python 3.13 --seed "$VENV"
# Pascal/Volta/Turing support: torch 2.6.0 built for CUDA 12.4 includes sm_60/sm_70/sm_75 kernels.
"${UVPIP[@]}" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
# torch 2.6 requires sympy==1.13.1: drop torch/torchvision/sympy pins from both locks
grep -vE '^(torch|torchvision|sympy)==' requirements/models.lock > /tmp/models-nottorch.lock
grep -vE '^sympy==' requirements/runtime.lock > /tmp/runtime-nosympy.lock
# wrapt: harmless extra so a host sitecustomize that still leaks in cannot abort the interpreter
"${UVPIP[@]}" -r /tmp/runtime-nosympy.lock -r /tmp/models-nottorch.lock sympy==1.13.1 wrapt
"${UVPIP[@]}" --no-deps -e .
"${PY[@]}" - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda build", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
for i in range(torch.cuda.device_count()):
    x = torch.randn(512, 512, device=f"cuda:{i}"); y = x @ x; torch.cuda.synchronize(i)
    print(f"gpu{i}", torch.cuda.get_device_name(i), "matmul ok")
print("arch list", torch.cuda.get_arch_list())
PY
"${PY[@]}" scripts/fetch_models.py --only rtdetr_r18 dinov2

mapfile -t EXPORTS < <(find "$DATA_ROOT" -name _annotations.coco.json -path '*/train/*' -printf '%h\n' | xargs -n1 dirname | sort -u)
echo "exports: ${EXPORTS[*]}"
[ "${#EXPORTS[@]}" -gt 0 ] || { echo "no COCO exports found under $DATA_ROOT"; exit 2; }
mkdir -p "$OUT"
"${PY[@]}" -m pvs.training.real5star prepare --exports "${EXPORTS[@]}" --out "$OUT/data" | tee "$OUT/prepare.log"

BATCH=$((BATCH_PER_GPU * NGPU))
echo "training on $NGPU GPU(s): global batch $BATCH ($BATCH_PER_GPU per GPU), $WORKERS data workers per GPU"
if [ "$NGPU" -gt 1 ]; then
  LAUNCH=("${PY[@]}" -m torch.distributed.run --standalone --nproc_per_node="$NGPU")
else
  LAUNCH=("${PY[@]}")
fi
# lr scaled with sqrt(global batch / 16) from 2e-4
LR=$("${PY[@]}" -c "print(round(2e-4 * ($BATCH / 16) ** 0.5, 6))")
"${LAUNCH[@]}" -m pvs.training.real5star train --data "$OUT/data" --run "$OUT" --device cuda --epochs "$EPOCHS" \
    --batch "$BATCH" --size 640 --lr "$LR" --workers "$WORKERS" 2>&1 | grep -v -i -E 'warning|mismatch|detach' | tee "$OUT/train.log"
[ -f "$OUT/train_report.json" ] || { echo "training failed, see $OUT/train.log"; exit 4; }
"${PY[@]}" -m pvs.training.real5star evaluate --data "$OUT/data" --run "$OUT" --device cuda --size 640 \
    --refs-per-class 40 2>&1 | grep -v -i warning | tee "$OUT/eval.log"
cp "$OUT/data/stats.json" "$OUT/data_stats.json"
tar -czf "$OUT/../pvs-run-artifacts.tgz" -C "$OUT" train_report.json eval_report.json data_stats.json \
    rtdetr_r18_package.onnx rtdetr_r18_package.json gallery_5star.npz prepare.log train.log eval.log
sha256sum "$OUT/../pvs-run-artifacts.tgz"
echo "DONE"
