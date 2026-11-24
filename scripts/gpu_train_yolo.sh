#!/usr/bin/env bash
# Laptop model: YOLO26n-seg (detection + segmentation + brand) + PP-OCRv5 mobile, trained and checked in ONE
# invocation on Kaggle (2 x T4). Same pattern as gpu_train_retail.sh / gpu_train_brands.sh.
#
#   ROBOFLOW_API_KEY=... [HF_TOKEN=...] REPO=... WORK=... OUT=... bash scripts/gpu_train_yolo.sh
#
# Stages: setup -> data (Roboflow projects in configs/brand_datasets.yaml + Open Food Facts close-ups) -> prepare
# (dedupe, leakage-safe splits) -> masks (SAM2 from the boxes, one shard per GPU) -> smoke check of the whole chain
# -> full training (time-boxed to the Kaggle session) -> NMS-free ONNX export (640 / 480 px, fp32 + fp16) ->
# test mAP (Ultralytics) + test evaluation with the exact laptop code (brand threshold chosen on validation) ->
# latency + memory on the T4 and on the CPU -> the live v2 site on the same test split -> upload.
# The laptop bundle ends up in $OUT/bundle (and on Hugging Face under bundle/ with HF_TOKEN).
# Ultralytics YOLO is AGPL-3.0: see docs/EDGE.md.
# Optional: EPOCHS (80), IMGSZ (640), BATCH (64 over both GPUs), SKIP_SMOKE=1, HF_REPO (pack-station-training),
#           TIME_LIMIT_H (12), PUSH_EVERY_S (240), MAX_PLAIN (15000), OFF_MAX_PER_BRAND (1500), OFF_MAX_NEG (7000)
set -euo pipefail
START_TS=${SESSION_START:-$(date +%s)}   # scripts/kaggle_autorun.py passes the session start on a retry
REPO=${REPO:-$HOME/product-vision-station}
WORK=${WORK:-/tmp/pvs-yolo}
OUT=${OUT:-$HOME/pvs-yolo-out}
EPOCHS=${EPOCHS:-80}
IMGSZ=${IMGSZ:-640}
BATCH=${BATCH:-64}
SKIP_SMOKE=${SKIP_SMOKE:-0}
HF_REPO=${HF_REPO:-pack-station-training}
TIME_LIMIT_H=${TIME_LIMIT_H:-12}
PUSH_EVERY_S=${PUSH_EVERY_S:-240}
MAX_PLAIN=${MAX_PLAIN:-15000}
OFF_MAX_PER_BRAND=${OFF_MAX_PER_BRAND:-1500}
OFF_MAX_NEG=${OFF_MAX_NEG:-7000}
LIST=configs/brand_datasets.yaml
# resume: a stage that finished in an earlier attempt of this Kaggle session (same $WORK) is skipped
is_done() { [ -f "$WORK/.done-$1" ]; }
mark_done() { touch "$WORK/.done-$1"; }
[ -n "${ROBOFLOW_API_KEY:-}" ] || { echo "ROBOFLOW_API_KEY is not set"; exit 2; }
mkdir -p "$OUT/logs" "$OUT/status" "$OUT/yolo" "$WORK"
LOG="$OUT/logs/run.log"
exec > >(tee -a "$LOG") 2>&1
STATUS="$OUT/status"
STAGE="setup"
WATCH_PID=""

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
NGPU=$(nvidia-smi -L | wc -l)
$SUDO apt-get update -qq >/dev/null 2>&1 || true
$SUDO apt-get install -y -qq git unzip curl libgl1 libglib2.0-0 >/dev/null 2>&1 || true
[ -d "$REPO" ] || git clone --depth 1 https://github.com/Rohitpro1/product-vision-station "$REPO"
cd "$REPO"
echo "repo commit: $(git log --oneline -1 2>/dev/null || echo unknown)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
UV_BIN=$(dirname "$(PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" command -v uv)")

# three venvs: models (SAM2, data tools), yolo (Ultralytics, AGPL), edge (what the laptop runs: no torch)
VENV="$REPO/.venv-gpu"; VENV_Y="$REPO/.venv-yolo"; VENV_E="$REPO/.venv-edge"; VENV_RF="$REPO/.venv-rf"
CLEAN=(env -i HOME="$HOME" LANG=C.UTF-8 TERM=dumb PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
       PATH="$VENV/bin:$UV_BIN:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
       LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" TMPDIR="${TMPDIR:-/tmp}" ROBOFLOW_API_KEY="$ROBOFLOW_API_KEY"
       HF_TOKEN="${HF_TOKEN:-}" PVS_STATUS_DIR="$STATUS" PVS_DISK_PATH="$WORK" YOLO_CONFIG_DIR="$WORK/ultralytics-cfg"
       HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4)
PY=("${CLEAN[@]}" "$VENV/bin/python" -s)
pyst() { local s=$1; shift; "${CLEAN[@]}" PVS_STAGE="$s" "$VENV/bin/python" -s "$@"; }
yst()  { local s=$1; shift; "${CLEAN[@]}" PVS_STAGE="$s" "$VENV_Y/bin/python" -s "$@"; }
est()  { local s=$1; shift; "${CLEAN[@]}" PVS_STAGE="$s" "$VENV_E/bin/python" -s "$@"; }
ev() { "${PY[@]}" -m pvs.training.progress "$@" || true; }
FILTER='warning|mismatch|detach|Loading weights|Duplicate provider|it/s\]'
filt() { grep --line-buffered -v -i -E "$FILTER" || true; }
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

uvi() { local v=$1; shift; "${CLEAN[@]}" uv pip install -q --python "$v/bin/python" "$@"; }
for v in "$VENV" "$VENV_Y" "$VENV_E" "$VENV_RF"; do [ -x "$v/bin/python" ] || "${CLEAN[@]}" uv venv -q --python 3.13 --seed "$v"; done
uvi "$VENV" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
grep -vE '^(torch|torchvision|sympy)==' requirements/models.lock > /tmp/models-nottorch.lock
grep -vE '^sympy==' requirements/runtime.lock > /tmp/runtime-nosympy.lock
uvi "$VENV" -r /tmp/runtime-nosympy.lock -r /tmp/models-nottorch.lock sympy==1.13.1 wrapt onnx pyarrow requests
uvi "$VENV" --no-deps -e .
uvi "$VENV_Y" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
uvi "$VENV_Y" ultralytics onnx onnxslim onnxruntime rapidocr requests scipy pyyaml
uvi "$VENV_Y" --no-deps -e .
uvi "$VENV_E" -r requirements/edge.txt "onnxruntime-gpu[cuda,cudnn]" scipy
uvi "$VENV_E" --no-deps -e .
uvi "$VENV_RF" roboflow==1.7.3
uvi "$VENV_RF" --no-deps -e .
"${CLEAN[@]}" uv cache clean -q 2>/dev/null || true   # wheels are installed; the cache only takes disk
df -h "$WORK" /kaggle/working 2>/dev/null || df -h "$WORK"
rm -f "$STATUS/.stop"; is_done setup-status || { rm -f "$STATUS/status.json"; mark_done setup-status; }
ev setup start
"${PY[@]}" -c "import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__);assert torch.cuda.is_available()"
"${CLEAN[@]}" "$VENV_Y/bin/python" -s -c "import ultralytics;print('ultralytics',ultralytics.__version__)"
"${CLEAN[@]}" "$VENV_E/bin/python" -s -c "import onnxruntime as o;o.preload_dlls() if hasattr(o,'preload_dlls') else None;print('onnxruntime',o.__version__,o.get_available_providers())"
"${PY[@]}" scripts/fetch_models.py --only sam2 rtdetr_r18 dinov2
[ -f "$WORK/live/release.json" ] || "${PY[@]}" -m pvs.training.brands3 fetch-live --out "$WORK/live"
DEV=$(seq -s, 0 $((NGPU - 1)))
# GPU preflight (~3 min): everything the smoke check would only reach after the downloads
if ! is_done preflight; then
mkdir -p "$WORK/pre"
(cd "$WORK/pre" && yst preflight -c "
from ultralytics import YOLO
m = YOLO('yolo26n-seg.pt')
m.train(data='coco8-seg.yaml', epochs=1, imgsz=320, batch=8, device='$DEV', workers=2, project='$WORK/pre',
        name='t', exist_ok=True, plots=False, verbose=False)
p = YOLO('$WORK/pre/t/weights/best.pt').export(format='onnx', imgsz=320, end2end=True, half=True, device=0)
print('PREFLIGHT_ONNX', p)" 2>&1 | filt | grep -E "PREFLIGHT|Error|error" || true)
PRE_ONNX=$(ls "$WORK"/pre/t/weights/best.onnx 2>/dev/null || true)
[ -n "$PRE_ONNX" ] || { echo "preflight: Ultralytics train/export failed"; exit 7; }
est preflight -c "
import numpy as np
from pvs.edge.seg import YoloSeg
s = YoloSeg('$PRE_ONNX', device='cuda'); s(np.zeros((480, 640, 3), np.uint8))
assert s.provider.startswith('CUDA'), s.provider
print('PREFLIGHT edge runtime on', s.provider, 'fp16', s.fp16)"
pyst preflight -c "
import json, numpy as np
from pvs.detectors.hf_models import Sam2MaskProvider
m = json.load(open('models/manifest.json'))['models']['sam2']
r = Sam2MaskProvider(m['repo'], m['revision'], device='cuda').masks(np.full((240, 320, 3), 200, np.uint8), [[40, 40, 200, 160]])
print('PREFLIGHT sam2 on cuda ok', len(r))"
echo "=== GPU preflight passed"
mark_done preflight
fi
"${PY[@]}" -m pvs.training.hub init --repo "$HF_REPO"
"${PY[@]}" -m pvs.training.hub watch --repo "$HF_REPO" --status-dir "$STATUS" --every "$PUSH_EVERY_S" &
WATCH_PID=$!
ev setup done

STAGE=download
if is_done download; then echo "=== download: done in an earlier attempt, skipped"; else
ev download start
(
  "${PY[@]}" -c "from huggingface_hub import hf_hub_download as d; print(d('openfoodfacts/product-database', 'food.parquet', repo_type='dataset', local_dir='$WORK/off'))"
  pyst off -m pvs.training.off scan --parquet "$WORK/off/food.parquet" --out "$WORK/off/products.json" > "$WORK/off/scan.log" 2>&1
  rm -f "$WORK/off/food.parquet"
) &
OFF_PID=$!
"${CLEAN[@]}" PVS_STAGE=download "$VENV_RF/bin/python" -s -m pvs.training.retail download --list "$LIST" --out "$WORK/rf" --fork
wait "$OFF_PID" || { echo "Open Food Facts scan failed:"; tail -n 30 "$WORK/off/scan.log" || true; exit 5; }
tail -30 "$WORK/off/scan.log"; cp "$WORK/off/scan.log" "$OUT/logs/off_scan.log"
"${PY[@]}" -c "import json;d=json.load(open('$WORK/off/products.json'));json.dump(d['stats'],open('$OUT/off_scan_stats.json','w'),indent=1)"
pyst off -m pvs.training.off download --products "$WORK/off/products.json" --exports "$WORK/rf" \
    --max-per-brand "$OFF_MAX_PER_BRAND" --max-neg "$OFF_MAX_NEG" --workers 12
cp "$WORK/rf/off_download_stats.json" "$WORK/rf/download_report.json" "$OUT/"
ev download done; mark_done download
fi

STAGE=prepare
if is_done prepare; then echo "=== prepare: done in an earlier attempt, skipped"; else
ev prepare start
pyst prepare -m pvs.training.retail prepare --exports "$WORK/rf" --list "$LIST" --out "$WORK/data"
cp "$WORK/data/stats.json" "$OUT/data_stats.json"
# the raw exports are no longer needed (prepare copied what it uses into $WORK/data): free their disk
find "$WORK/rf" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
df -h "$WORK"
ev prepare done; mark_done prepare
fi

STAGE=masks
D="$WORK/yolo-data"
if is_done masks; then echo "=== masks: done in an earlier attempt, skipped"; else
ev masks start
rm -rf "$D"
MASK_PIDS=()
for g in $(seq 0 $((NGPU - 1))); do
  "${CLEAN[@]}" CUDA_VISIBLE_DEVICES="$g" PVS_STAGE=masks "$VENV/bin/python" -s -m pvs.training.yolo dataset \
      --data "$WORK/data" --out "$D" --max-plain "$MAX_PLAIN" --shard "$g" --shards "$NGPU" --device cuda \
      > "$OUT/logs/masks_$g.log" 2>&1 &
  MASK_PIDS+=($!)
done
# wait for the mask shards only: a bare `wait` would also wait for the never-ending status uploader
for p in "${MASK_PIDS[@]}"; do
  wait "$p" || { echo "mask generation failed:"; tail -n 25 "$OUT"/logs/masks_*.log || true; exit 6; }
done
pyst masks -m pvs.training.yolo finalize --out "$D" | sed -n 1,60p
cp "$D/dataset_report.json" "$OUT/yolo/"
ev masks done; mark_done masks
fi

# chain <prefix> <outdir> <epochs> <imgsz> <hours or 0> <eval limit> <extra train args...>
chain() {
  local P=$1 O=$2 EP=$3 SZ=$4 H=$5 LIM=$6; shift 6
  mkdir -p "$O"
  df -h "$WORK" | tail -1
  STAGE=${P}train
  local K=${P:-full}; K=${K%:}
  if is_done "$K-train"; then echo "=== $K train: done in an earlier attempt, skipped"; else
  ev "$STAGE" start
  local TH=(); [ "$H" != "0" ] && TH=(--time-h "$H")
  local MODEL=(); if [ -f "$O/run/train/weights/last.pt" ]; then   # an attempt died mid-training: continue from it
    cp "$O/run/train/weights/last.pt" "$WORK/resume-$K.pt"; MODEL=(--model "$WORK/resume-$K.pt")
    echo "=== $K train: continuing from the last checkpoint of the earlier attempt"; fi
  (cd "$WORK" && yst "${P}train" -m pvs.training.yolo train --yaml "$D/data.yaml" --out "$O/run" --epochs "$EP" \
      --imgsz "$SZ" --batch "$BATCH" --device "$DEV" --workers 4 "${TH[@]}" "${MODEL[@]}" "$@" 2>&1 | filt)
  [ -f "$O/run/train/weights/best.pt" ] || { echo "training produced no weights"; exit 4; }
  cp "$O/run/train_report.json" "$O/run/train/results.csv" "$O/" 2>/dev/null || true
  ev "$STAGE" done; mark_done "$K-train"
  fi
  STAGE=${P}export
  if is_done "$K-export"; then echo "=== $K export: done in an earlier attempt, skipped"; else
  ev "$STAGE" start
  local B="$O/bundle"; rm -rf "$B"
  (cd "$WORK" && yst "${P}export" -m pvs.training.yolo export --weights "$O/run/train/weights/best.pt" --out "$B" \
      --imgsz 640 480 --default-size 480 --fp16 --device 0 2>&1 | filt | tail -5)
  [ -f "$B/edge.json" ] || { echo "export produced no bundle"; exit 8; }
  if [ -z "$P" ]; then rm -rf "$OUT/bundle"; cp -r "$B" "$OUT/bundle"; fi   # uploaded even if a check below fails
  ev "$STAGE" done; mark_done "$K-export"
  fi
  local B="$O/bundle"
  STAGE=${P}test; ev "$STAGE" start
  (cd "$WORK" && yst "${P}test" -m pvs.training.yolo val --weights "$O/run/train/weights/best.pt" --yaml "$D/data.yaml" \
      --out "$O/yolo_val_test.json" --imgsz "$SZ" --device 0 2>&1 | filt | tail -20) || echo "test mAP failed (not fatal)"
  est "${P}test" -m pvs.training.yolo evaluate --data "$D" --bundle "$B" --out "$O/yolo_report.json" --device cuda \
      --limit "$LIM" --ocr-sample $([ "$P" = "smoke:" ] && echo 20 || echo 400) 2>&1 | filt | tail -40 \
      || { echo "evaluation failed"; [ -z "$P" ] || exit 9; }
  ev "$STAGE" done
  STAGE=${P}bench; ev "$STAGE" start
  ls "$D/images/test" | sed -n "1,$([ "$P" = "smoke:" ] && echo 20 || echo 200)p" | sed "s#^#$D/images/test/#" > "$WORK/bench_list.txt"
  rm -rf "$WORK/bench_imgs"; mkdir -p "$WORK/bench_imgs"; xargs -a "$WORK/bench_list.txt" cp -L -t "$WORK/bench_imgs"
  est "${P}bench" -m pvs.edge.station bench --bundle "$B" --images "$WORK/bench_imgs" --device cuda \
      --out "$O/bench_t4_gpu.json" 2>&1 | filt | tail -25 || echo "GPU bench failed (not fatal)"
  est "${P}bench" -m pvs.edge.station bench --bundle "$B" --images "$WORK/bench_imgs" --device cpu --threads 4 \
      --mem-limit-gb 4 --out "$O/bench_cpu_4gb.json" 2>&1 | filt | tail -25 || echo "CPU bench failed (not fatal)"
  ev "$STAGE" done
}

if [ "$SKIP_SMOKE" != "1" ] && ! is_done smoke; then
  chain "smoke:" "$WORK/smoke-out" 1 320 0 40 --fraction 0.03
  mark_done smoke
  echo "=== smoke check passed: the whole chain works; starting the full run"
fi
# time-box training: leave ~75 min for export, tests, benchmarks, the v2 comparison and upload
LEFT_H=$("${PY[@]}" -c "import time;print(round(max(0.5, ($START_TS + $TIME_LIMIT_H*3600 - time.time())/3600 - 1.25), 2))")
echo "training budget: $LEFT_H h"
chain "" "$OUT/yolo" "$EPOCHS" "$IMGSZ" "$LEFT_H" 3000
cp "$OUT/yolo/bundle/edge.json" "$OUT/bundle/edge.json"   # brand_conf chosen on validation by evaluate
(cd "$OUT/bundle" && find . -type f | sort | xargs sha256sum) > "$OUT/bundle_sha256.txt"

STAGE=compare; ev compare start   # what the live site (release v2) does on the same test split, end to end
mkdir -p "$OUT/e2e_v2"
pyst compare -m pvs.training.brands e2e --data "$WORK/data" --bundle "$WORK/live" --out "$OUT/e2e_v2" \
    --workers 4 --max-branded 300 --max-other 200 2>&1 | filt | tail -20 || echo "v2 comparison failed (not fatal)"
ev compare done

STAGE=upload; ev upload start
tar -czf "$OUT/yolo-edge-artifacts.tgz" -C "$OUT" data_stats.json download_report.json off_scan_stats.json \
    off_download_stats.json bundle_sha256.txt yolo/dataset_report.json yolo/train_report.json yolo/results.csv \
    yolo/yolo_val_test.json yolo/yolo_report.json yolo/bench_t4_gpu.json yolo/bench_cpu_4gb.json \
    e2e_v2/e2e_report.json logs 2>/dev/null || true
tar -czf "$OUT/edge_bundle.tgz" -C "$OUT" bundle
sha256sum "$OUT/yolo-edge-artifacts.tgz" "$OUT/edge_bundle.tgz"
ev upload done
echo "DONE"
