#!/usr/bin/env bash
# Run 3: chocolate-brand accuracy. ONE invocation on Kaggle (T4 GPU), same pattern as gpu_train_retail.sh.
#
#   ROBOFLOW_API_KEY=... [HF_TOKEN=...] REPO=... WORK=... OUT=... bash scripts/gpu_train_brands.sh
#
# What it does:
#   1. downloads the live release (release.json) = the baseline to beat; the package detector is reused as is
#   2. Roboflow projects (configs/brand_datasets.yaml) and, in parallel, the Open Food Facts product database
#      scan (front-of-pack close-ups, labelled by brand tags + product names); then the photos
#   3. prepare (dedupe + leakage-safe splits across every source)
#   4. smoke check of the whole chain on a tiny budget, then the full run:
#      crops -> fine-tune DINOv2 (CosFace) -> evaluate (frozen vs fine-tuned, chosen on VALIDATION; live v2 scored
#      on the same TEST crops) -> v3 bundle -> external photos (v2 and v3) -> end to end (v2 and v3) -> decision
#   5. uploads reports + the v3 bundle to https://huggingface.co/<you>/$HF_REPO (needs HF_TOKEN)
# Nothing is deployed by this script: release.json is changed by hand only if decision.json says "deploy v3".
# Optional: STEPS (5000), BATCH (96), WORKERS (4), SKIP_SMOKE=1, HF_REPO (pack-station-training),
#           TIME_LIMIT_H (12), PUSH_EVERY_S (240), OFF_MAX_PER_BRAND (1500), OFF_MAX_NEG (7000)
set -euo pipefail
START_TS=$(date +%s)
REPO=${REPO:-$HOME/product-vision-station}
WORK=${WORK:-/tmp/pvs-brands}
OUT=${OUT:-$HOME/pvs-brands-out}
STEPS=${STEPS:-5000}
BATCH=${BATCH:-96}
WORKERS=${WORKERS:-4}
SKIP_SMOKE=${SKIP_SMOKE:-0}
HF_REPO=${HF_REPO:-pack-station-training}
TIME_LIMIT_H=${TIME_LIMIT_H:-12}
PUSH_EVERY_S=${PUSH_EVERY_S:-240}
OFF_MAX_PER_BRAND=${OFF_MAX_PER_BRAND:-1500}
OFF_MAX_NEG=${OFF_MAX_NEG:-7000}
LIST=configs/brand_datasets.yaml
[ -n "${ROBOFLOW_API_KEY:-}" ] || { echo "ROBOFLOW_API_KEY is not set"; exit 2; }
mkdir -p "$OUT/logs" "$OUT/status" "$WORK"
LOG="$OUT/logs/run.log"
exec > >(tee -a "$LOG") 2>&1
STATUS="$OUT/status"
STAGE="setup"
WATCH_PID=""

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
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
pyst() { local s=$1; shift; "${CLEAN[@]}" PVS_STAGE="$s" "$VENV/bin/python" -s "$@"; }
ev() { "${PY[@]}" -m pvs.training.progress "$@" || true; }   # stage bookkeeping (never fatal)
FILTER='warning|mismatch|detach|Loading weights|Duplicate provider'
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

"${CLEAN[@]}" uv venv -q --python 3.13 --seed "$VENV"
"${UVPIP[@]}" --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
grep -vE '^(torch|torchvision|sympy)==' requirements/models.lock > /tmp/models-nottorch.lock
grep -vE '^sympy==' requirements/runtime.lock > /tmp/runtime-nosympy.lock
"${UVPIP[@]}" -r /tmp/runtime-nosympy.lock -r /tmp/models-nottorch.lock sympy==1.13.1 wrapt onnx pyarrow requests
"${UVPIP[@]}" --no-deps -e .
VENV_RF="$REPO/.venv-rf"   # roboflow needs typer<0.26; keep it out of the model venv
"${CLEAN[@]}" uv venv -q --python 3.13 --seed "$VENV_RF"
"${CLEAN[@]}" uv pip install -q --python "$VENV_RF/bin/python" roboflow==1.7.3
"${CLEAN[@]}" uv pip install -q --python "$VENV_RF/bin/python" --no-deps -e .
rm -f "$STATUS/.stop" "$STATUS/status.json"
ev setup start
"${PY[@]}" -c "import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__);assert torch.cuda.is_available();print('gpu',torch.cuda.get_device_name(0))"
"${PY[@]}" scripts/fetch_models.py --only dinov2
"${PY[@]}" -m pvs.training.brands3 fetch-live --out "$WORK/live"
"${PY[@]}" -m pvs.training.hub init --repo "$HF_REPO"
"${PY[@]}" -m pvs.training.hub watch --repo "$HF_REPO" --status-dir "$STATUS" --every "$PUSH_EVERY_S" &
WATCH_PID=$!
ev setup done

# Roboflow download and the Open Food Facts scan run side by side; OFF photos are fetched afterwards because
# both write download_report.json (Roboflow rewrites it, OFF appends to it)
STAGE=download; ev download start
(
  "${PY[@]}" -c "from huggingface_hub import hf_hub_download as d; print(d('openfoodfacts/product-database', 'food.parquet', repo_type='dataset', local_dir='$WORK/off'))"
  pyst off -m pvs.training.off scan --parquet "$WORK/off/food.parquet" --out "$WORK/off/products.json" > "$WORK/off/scan.log" 2>&1
  rm -f "$WORK/off/food.parquet"
) &
OFF_PID=$!
"${CLEAN[@]}" PVS_STAGE=download "$VENV_RF/bin/python" -s -m pvs.training.retail download --list "$LIST" --out "$WORK/rf"
wait "$OFF_PID" || { echo "Open Food Facts scan failed:"; tail -30 "$WORK/off/scan.log"; exit 5; }
tail -40 "$WORK/off/scan.log"
cp "$WORK/off/scan.log" "$OUT/logs/off_scan.log"
"${PY[@]}" -c "import json;d=json.load(open('$WORK/off/products.json'));json.dump(d['stats'],open('$OUT/off_scan_stats.json','w'),indent=1)"
pyst off -m pvs.training.off download --products "$WORK/off/products.json" --exports "$WORK/rf" \
    --max-per-brand "$OFF_MAX_PER_BRAND" --max-neg "$OFF_MAX_NEG" --workers 12
cp "$WORK/rf/off_download_stats.json" "$WORK/rf/download_report.json" "$OUT/"
ev download done

STAGE=prepare; ev prepare start
pyst prepare -m pvs.training.retail prepare --exports "$WORK/rf" --list "$LIST" --out "$WORK/data"
cp "$WORK/data/stats.json" "$OUT/data_stats.json"
ev prepare done

# leave ~70 min after training for evaluate + bundle + checks + upload inside the session limit
STOP_AT=$(( START_TS + TIME_LIMIT_H * 3600 - 80 * 60 ))

# chain <prefix> <outdir> <steps> <batch> <eval-every> <crops extra...>
chain() {
  local P=$1 O=$2 ST=$3 B=$4 EE=$5; shift 5
  local E2EB=300 E2EO=200 TX=()
  if [ "$P" = "smoke:" ]; then E2EB=10; E2EO=6; TX=(--probe-refs 20 --probe-negs 200 --probe-val 300); rm -rf "$O"; fi
  mkdir -p "$O"
  STAGE=${P}crops; ev "$STAGE" start
  pyst "${P}crops" -m pvs.training.brands3 crops --data "$WORK/data" --out "$O/crops" "$@" 2>&1 | filt
  cp "$O/crops/crop_stats.json" "$O/"
  ev "$STAGE" done
  STAGE=${P}train; ev "$STAGE" start
  pyst "${P}train" -m pvs.training.brands3 train --crops "$O/crops" --out "$O/ft" --device cuda:0 --steps "$ST" \
      --batch "$B" --workers "$WORKERS" --eval-every "$EE" --stop-at "$STOP_AT" "${TX[@]}" 2>&1 | filt
  [ -f "$O/ft/train_report.json" ] || { echo "training produced no report"; exit 4; }
  ev "$STAGE" done
  STAGE=${P}evaluate; ev "$STAGE" start
  pyst "${P}evaluate" -m pvs.training.brands3 evaluate --crops "$O/crops" --ft "$O/ft" --v2 "$WORK/live" \
      --out "$O/brands" --device cuda:0 2>&1 | filt
  ev "$STAGE" done
  STAGE=${P}bundle; ev "$STAGE" start
  pyst "${P}bundle" -m pvs.training.brands3 bundle --base "$WORK/live" --brands "$O/brands" --out "$O/bundle" \
      > /dev/null
  (cd "$O/bundle" && find . -type f | sort | xargs sha256sum) > "$O/bundle_sha256.txt"
  ev "$STAGE" done
  STAGE=${P}checks; ev "$STAGE" start
  for v in v2:"$WORK/live" v3:"$O/bundle"; do
    pyst "${P}checks" -m pvs.training.brands3 external --bundle "${v#*:}" --cache "$WORK/external" \
        --out "$O/external_${v%%:*}.json" 2>&1 | filt | tail -8
  done
  pyst "${P}checks" -m pvs.training.brands e2e --data "$WORK/data" --bundle "$O/bundle" --out "$O/brands" \
      --workers 4 --max-branded "$E2EB" --max-other "$E2EO" 2>&1 | filt | tail -25
  mkdir -p "$O/e2e_v2"
  pyst "${P}checks" -m pvs.training.brands e2e --data "$WORK/data" --bundle "$WORK/live" --out "$O/e2e_v2" \
      --workers 4 --max-branded "$E2EB" --max-other "$E2EO" 2>&1 | filt | tail -25
  "${PY[@]}" -m pvs.training.brands3 decide --report "$O/brands/brands_report.json" --ext-v2 "$O/external_v2.json" \
      --ext-v3 "$O/external_v3.json" --out "$O/decision.json"
  ev "$STAGE" done
}

if [ "$SKIP_SMOKE" != "1" ]; then
  chain "smoke:" "$WORK/smoke-out" 20 16 10 --max-crops 1500
  echo "=== smoke check passed: the whole chain works; starting the full run"
fi
chain "" "$OUT" "$STEPS" "$BATCH" 500

STAGE=upload; ev upload start
tar -czf "$OUT/brands-v3-artifacts.tgz" -C "$OUT" data_stats.json download_report.json off_scan_stats.json \
    off_download_stats.json crop_stats.json bundle_sha256.txt ft/train_report.json brands/brands_report.json \
    brands/e2e_report.json e2e_v2/e2e_report.json brands/thresholds_brands.yaml brands/catalogue_brands.json \
    external_v2.json external_v3.json decision.json logs
sha256sum "$OUT/brands-v3-artifacts.tgz" "$OUT/bundle/models/trained/"*
cat "$OUT/decision.json"
ev upload done
echo "DONE"
