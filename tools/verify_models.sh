#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT/models"
MANIFEST="$MODEL_DIR/SHA256SUMS"

if [[ ! -f "$MANIFEST" ]]; then
  echo "error: missing model checksum manifest: $MANIFEST" >&2
  exit 1
fi

if find "$MODEL_DIR" -type l -print -quit | grep -q .; then
  echo "error: managed model directory contains a symbolic link" >&2
  find "$MODEL_DIR" -maxdepth 1 -type l -print >&2
  exit 1
fi

verified=0
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  expected="${line%%  *}"
  filename="${line#*  }"
  path="$MODEL_DIR/$filename"

  if [[ ! -f "$path" ]]; then
    echo "error: missing managed model: $filename" >&2
    exit 1
  fi

  actual="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "error: checksum mismatch: $filename" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
  verified=$((verified + 1))
done < "$MANIFEST"

if [[ "$verified" -eq 0 ]]; then
  echo "error: model checksum manifest is empty" >&2
  exit 1
fi

echo "✓ verified $verified managed model files (no external symlinks)"

