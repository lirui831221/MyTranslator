#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/bin/qwen-asr-dist"
OUTPUT="$DEST/qwen-asr-worker"
BUILD_ROOT="$ROOT/.build/qwen-asr-worker"
VENV="$BUILD_ROOT/venv"

if [[ -x "$OUTPUT" && "${MYT_REBUILD_QWEN_ASR:-0}" != "1" ]]; then
  echo "✓ Qwen3-ASR worker already built"
  exit 0
fi

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" ]]; then
  echo "error: uv is required to build the Qwen3-ASR worker" >&2
  exit 1
fi

mkdir -p "$BUILD_ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV_BIN" venv --python 3.13 "$VENV"
fi
"$UV_BIN" pip install --python "$VENV/bin/python" \
  "mlx-qwen3-asr==0.4.4" "pyinstaller==6.16.0"

STAGE="$(mktemp -d /tmp/myt-qwen-asr-dist.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
"$VENV/bin/pyinstaller" \
  --noconfirm \
  --clean \
  --onedir \
  --name qwen-asr-worker \
  --collect-all mlx \
  --collect-all mlx_qwen3_asr \
  --distpath "$STAGE/dist" \
  --workpath "$STAGE/work" \
  --specpath "$STAGE" \
  "$ROOT/native/qwen-asr-worker.py"

rm -rf "$DEST"
mv "$STAGE/dist/qwen-asr-worker" "$DEST"
chmod +x "$OUTPUT"
echo "✓ built Qwen3-ASR worker: $OUTPUT"
du -sh "$DEST" | awk '{print "  runtime size:", $1}'
