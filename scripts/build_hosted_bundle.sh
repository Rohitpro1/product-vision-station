#!/usr/bin/env bash
# Builds the runtime bundle the hosted API (deploy/vercel-api) downloads on a cold start:
# trained ONNX detector + hosted config + thresholds + demo catalogue + model manifest.
# Usage: bash scripts/build_hosted_bundle.sh path/to/rtdetr_r18_package.onnx [out.tgz]
set -euo pipefail
ONNX=${1:?path to rtdetr_r18_package.onnx}
OUT=${2:-dist/pvs-hosted-bundle.tgz}
EXPECT=68782ed8d550fa47334512634239eec067a032063e85ea7ef1fcb697560cde06   # Kaggle 2xT4 run export
echo "$EXPECT  $ONNX" | sha256sum -c -
cd "$(dirname "$0")/.."
STAGE=$(mktemp -d)
mkdir -p "$STAGE/models/trained" "$STAGE/configs" "$(dirname "$OUT")"
cp "$ONNX" "$STAGE/models/trained/rtdetr_r18_package.onnx"
cp configs/station.hosted.yaml configs/thresholds.yaml "$STAGE/configs/"
cp models/manifest.json "$STAGE/models/"
cp -r catalogues "$STAGE/"
# reproducible archive: fixed owner, mtime and ordering
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-11-21 00:00Z' -C "$STAGE" -cf - . | gzip -n -9 > "$OUT"
rm -rf "$STAGE"
sha256sum "$OUT"
