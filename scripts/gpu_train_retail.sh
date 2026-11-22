#!/usr/bin/env bash
# Retail detector + chocolate-brand gallery, end to end, in ONE invocation (built for Kaggle 2 x T4).
#
#   ROBOFLOW_API_KEY=... [HF_TOKEN=...] REPO=... WORK=... OUT=... bash scripts/gpu_train_retail.sh [EPOCHS]
#
# Order: setup -> download -> prepare -> smoke check (whole chain on a tiny budget) -> full run -> publish.
# Live progress: every stage prints "[stage] done/total | rate | stage ETA (IST) | run ETA (IST)" lines and keeps
# $OUT/status/status.json. With HF_TOKEN (write token) the run also keeps a live progress page at
# https://huggingface.co/<you>/$HF_REPO (private) and uploads reports + bundle there at the end, or the log tail
# if it fails.
# Optional: EPOCHS (arg, default 20), NGPU, BATCH_PER_GPU (16), WORKERS (4), E2E_WORKERS (4), SKIP_SMOKE=1,
#           HF_REPO (pack-station-training), TIME_LIMIT_H (12 = Kaggle session limit), PUSH_EVERY_S (240)
set -euo pipefail
START_TS=$(date +%s)
EPOCHS=${1:-20}
REPO=${REPO:-$HOME/product-vision-station}
WORK=${WORK:-/tmp/pvs-retail}
OUT=${OUT:-$HOME/pvs-retail-out}
BATCH_PER_GPU=${BATCH_PER_GPU:-16}
WORKERS=${WORKERS:-4}
E2E_WORKERS=${E2E_WORKERS:-4}
SKIP_SMOKE=${SKIP_SMOKE:-0}
HF_REPO=${HF_REPO:-pack-station-training}
TIME_LIMIT_H=${TIME_LIMIT_H:-12}
PUSH_EVERY_S=${PUSH_EVERY_S:-240}
[ -n "${ROBOFLOW_API_KEY:-}" ] || { echo "ROBOFLOW_API_KEY is not set"; exit 2; }
mkdir -p "$OUT/logs" "$OUT/status" "$WORK"
LOG="$OUT/logs/run.log"
exec > >(tee -a "$LOG") 2>&1
STATUS="$OUT/status"
STAGE="setup"
WATCH_PID=""

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
$SUDO apt-get update -qq >/dev/null 2>&1 || true
$SUDO apt-get install -y -qq git unzip curl libgl1 libglib2.0-0 >/dev/null 2>&1 || true
[ -d "$REPO" ] || git clone --depth 1 https://github.com/Rohitpro1/product-vision-station "$REPO"
cd "$REPO"
echo "repo commit: $(git log --oneline -1 2>/dev/null || echo unknown)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
UV_BIN=$(dirname "$(PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" command -v uv)")

# clean environment for every uv/python call (Kaggle injects PYTHONPATH + a sitecustomize that imports wrapt)
VENV="$REPO/.venv-gpu"
CLEAN=(env -i HOME="$HOME" LANG=C.UTF-8 TERM=dumb PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
       PATH="$VENV/bin:$UV_BIN:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
       LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" TMPDIR="${TMPDIR:-/tmp}" ROBOFLOW_API_KEY="$ROBOFLOW_API_KEY"
       HF_TOKEN="${HF_TOKEN:-}" PVS_STATUS_DIR="$STATUS"
       HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4)
PY=("${CLEAN[@]}" "$VENV/bin/python" -s)
UVPIP=("${CLEAN[@]}" uv pip install -q --python "$VENV/bin/python")
# run a python module under a named progress stage: pyst <stage> <args...>
pyst() { local s=$1; shift; "${CLEAN[@]}" PVS_STAGE="$s" "$VENV/bin/python" -s "$@"; }

ev() { "${PY[@]}" -m pvs.training.progress "$@" || true; }   # stage bookkeeping (never fatal)
on_exit() {
  rc=$?
  set +e
  if [ "$rc" -eq 0 ]; then ev "$STAGE" finished "all stages done"; else ev "$STAGE" failed "exit $rc in stage $STAGE"; fi
  if [ -x "$VENV/bin/python" ]; then
    "${PY[@]}" -m pvs.training.hub final --repo "$HF_REPO" --status-dir "$STATUS" --out "$OUT" --log "$LOG"
  fi
  touch "$STATUS/.stop"; [ -n "$WATCH_PID" ] && wait "$WATCH_PID" 2>/dev/null
  echo "total wall time: $(( ($(date +%s) - START_TS) / 60 )) min (exit $rc)"
}
trap on_exit EXIT

"${CLEAN[@]}" uv venv -q --python 3.13 --seed "$VENV"
"${UVPIP[@]}" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
grep -vE '^(torch|torchvision|sympy)==' requirements/models.lock > /tmp/models-nottorch.lock
grep -vE '^sympy==' requirements/runtime.lock > /tmp/runtime-nosympy.lock
"${UVPIP[@]}" -r /tmp/runtime-nosympy.lock -r /tmp/models-nottorch.lock sympy==1.13.1 wrapt onnx
"${UVPIP[@]}" --no-deps -e .
# roboflow (download only) gets its own venv: it needs typer<0.26 while the model stack pins typer 0.29, and
# inside the main venv the resolver silently fell back to roboflow 0.1.1 (no deps, no API)
VENV_RF="$REPO/.venv-rf"
"${CLEAN[@]}" uv venv -q --python 3.13 --seed "$VENV_RF"
"${CLEAN[@]}" uv pip install -q --python "$VENV_RF/bin/python" roboflow==1.7.3
"${CLEAN[@]}" uv pip install -q --python "$VENV_RF/bin/python" --no-deps -e .
"${CLEAN[@]}" "$VENV_RF/bin/python" -s -c "import roboflow, requests; print('roboflow', roboflow.__version__)"
rm -f "$STATUS/.stop" "$STATUS/status.json"
ev setup start
"${PY[@]}" -c "import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__);assert torch.cuda.is_available();[print('gpu',i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
"${PY[@]}" scripts/fetch_models.py --only rtdetr_r18 dinov2
"${PY[@]}" -m pvs.training.hub init --repo "$HF_REPO"
"${PY[@]}" -m pvs.training.hub watch --repo "$HF_REPO" --status-dir "$STATUS" --every "$PUSH_EVERY_S" &
WATCH_PID=$!
ev setup done

STAGE=download; ev download start
"${CLEAN[@]}" PVS_STAGE=download "$VENV_RF/bin/python" -s -m pvs.training.retail download --out "$WORK/rf"
ev download done
STAGE=prepare; ev prepare start
pyst prepare -m pvs.training.retail prepare --exports "$WORK/rf" --out "$WORK/data"
ev prepare done

BATCH=$((BATCH_PER_GPU * NGPU))
LR=$("${PY[@]}" -c "print(round(2e-4 * ($BATCH / 16) ** 0.5, 6))")
DIST=(); [ "$NGPU" -gt 1 ] && DIST=(-m torch.distributed.run --standalone --nproc_per_node="$NGPU")
# leave time after training for export + brands + e2e + upload (~75 min) inside the session limit (15 min margin)
STOP_AT=$(( START_TS + TIME_LIMIT_H * 3600 - 90 * 60 ))
FILTER='warning|mismatch|detach|Loading weights'
filt() { grep --line-buffered -v -i -E "$FILTER" || true; }   # never fails by itself; pipefail keeps python's exit code

# pipeline <stage prefix> <outdir> <run dir> <epochs> <extra train args...>
pipeline() {
  local P=$1 O=$2 RUN=$3 EP=$4; shift 4
  local EVAL=2000 E2EB=300 E2EO=200 BEXTRA=()
  if [ "$P" = "smoke:" ]; then EVAL=50; E2EB=12; E2EO=8; BEXTRA=(--max-crops 600 --min-train 3); rm -rf "$RUN"; fi
  mkdir -p "$O"
  STAGE=${P}train; ev "$STAGE" start
  echo "training on $NGPU GPU(s): global batch $BATCH, lr $LR, $EP epoch(s), stop-at $(date -d @"$STOP_AT" '+%H:%M UTC')"
  pyst "${P}train" "${DIST[@]}" -m pvs.training.real5star train --data "$WORK/data" --run "$RUN" --device cuda \
      --epochs "$EP" --batch "$BATCH" --size 640 --lr "$LR" --workers "$WORKERS" --val-limit 400 --stop-at "$STOP_AT" \
      "$@" 2>&1 | filt
  [ -f "$RUN/train_report.json" ] || { echo "training produced no report"; exit 4; }
  ev "$STAGE" done
  STAGE=${P}export; ev "$STAGE" start
  pyst "${P}export" -m pvs.training.retail export --data "$WORK/data" --run "$RUN" --out "$O/detector" \
      --test-limit "$EVAL" 2>&1 | filt
  cp "$RUN/train_report.json" "$O/detector/"
  ev "$STAGE" done
  STAGE=${P}brands; ev "$STAGE" start
  pyst "${P}brands" -m pvs.training.brands build --data "$WORK/data" --out "$O/brands" --device cuda "${BEXTRA[@]}" \
      2>&1 | filt
  ev "$STAGE" done
  STAGE=${P}bundle; ev "$STAGE" start
  local B="$O/bundle"; rm -rf "$B"; mkdir -p "$B/configs" "$B/models/trained" "$B/catalogues/brands"
  cp configs/station.hosted-brands.yaml "$B/configs/station.hosted.yaml"
  cp "$O/brands/thresholds_brands.yaml" "$B/configs/"
  cp "$O/brands/catalogue_brands.json" "$B/catalogues/brands/catalogue.json"
  cp "$O/detector/rtdetr_r18_package.onnx" "$O/brands/dinov2_small.onnx" "$O/brands/gallery_brands.npz" "$B/models/trained/"
  cp models/manifest.json "$B/models/"
  (cd "$B" && find . -type f | sort | xargs sha256sum) > "$O/bundle_sha256.txt"
  ev "$STAGE" done
  STAGE=${P}e2e; ev "$STAGE" start
  pyst "${P}e2e" -m pvs.training.brands e2e --data "$WORK/data" --bundle "$B" --out "$O/brands" \
      --workers "$E2E_WORKERS" --max-branded "$E2EB" --max-other "$E2EO" 2>&1 | filt
  ev "$STAGE" done
}

if [ "$SKIP_SMOKE" != "1" ]; then
  pipeline "smoke:" "$WORK/smoke-out" "$WORK/run-smoke" 1 --max-steps 30
  echo "=== smoke check passed: the whole chain works; starting the full run"
fi
pipeline "" "$OUT" "$WORK/run" "$EPOCHS"

STAGE=upload; ev upload start
cp "$WORK/data/stats.json" "$OUT/data_stats.json"; cp "$WORK/rf/download_report.json" "$OUT/"
tar -czf "$OUT/retail-v2-artifacts.tgz" -C "$OUT" data_stats.json download_report.json bundle_sha256.txt logs \
    detector/detector_report.json detector/train_report.json detector/rtdetr_r18_package.json \
    brands/brands_report.json brands/e2e_report.json brands/thresholds_brands.yaml brands/catalogue_brands.json
sha256sum "$OUT/retail-v2-artifacts.tgz"
du -sh "$OUT"/bundle/models/trained/*
ev upload done
echo "DONE"
