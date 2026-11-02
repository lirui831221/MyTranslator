#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/bin/qwen-translation-dist"
OUTPUT="$DEST/qwen-translation-worker"
BUILD_ROOT="$ROOT/.build/qwen-translation-worker"
VENV="$BUILD_ROOT/venv"

if [[ -x "$OUTPUT" \
  && "$OUTPUT" -nt "$ROOT/native/qwen-translation-worker.py" \
  && "${MYT_REBUILD_QWEN_TRANSLATION:-0}" != "1" ]]; then
  echo "✓ Qwen3.5 translation worker already built"
  exit 0
fi

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" ]]; then
  echo "error: uv is required to build the Qwen3.5 translation worker" >&2
  exit 1
fi

mkdir -p "$BUILD_ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV_BIN" venv --python 3.13 "$VENV"
fi
"$UV_BIN" pip install --python "$VENV/bin/python" \
  "mlx-vlm==0.7.10" "pyinstaller==6.16.0"

STAGE="$(mktemp -d /tmp/myt-qwen-translation-dist.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT
"$VENV/bin/pyinstaller" \
  --noconfirm \
  --clean \
  --onedir \
  --name qwen-translation-worker \
  --collect-all mlx \
  --collect-submodules mlx_vlm.models.qwen3_5 \
  --collect-submodules mlx_vlm.models.qwen3_vl \
  --collect-data mlx_vlm \
  --collect-data transformers \
  --exclude-module cv2 \
  --exclude-module datasets \
  --exclude-module pandas \
  --exclude-module pyarrow \
  --exclude-module scipy \
  --exclude-module sklearn \
  --exclude-module matplotlib \
  --exclude-module tkinter \
  --exclude-module sounddevice \
  --exclude-module gradio \
  --exclude-module fastapi \
  --exclude-module uvicorn \
  --distpath "$STAGE/dist" \
  --workpath "$STAGE/work" \
  --specpath "$STAGE" \
  "$ROOT/native/qwen-translation-worker.py"

rm -rf "$DEST"
mv "$STAGE/dist/qwen-translation-worker" "$DEST"
chmod +x "$OUTPUT"
echo "✓ built Qwen3.5 translation worker: $OUTPUT"
du -sh "$DEST" | awk '{print "  runtime size:", $1}'
