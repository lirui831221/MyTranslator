#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="$ROOT/native/system-audio-capture.swift"
OUTPUT="$ROOT/bin/system-audio-capture"
TEMP_OUTPUT="$ROOT/bin/.system-audio-capture.build"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: system audio helper requires macOS" >&2
  exit 1
fi

xcrun swiftc \
  -O \
  -whole-module-optimization \
  -swift-version 5 \
  -target arm64-apple-macosx13.0 \
  -framework ScreenCaptureKit \
  -framework AVFoundation \
  -framework CoreMedia \
  "$SOURCE" \
  -o "$TEMP_OUTPUT"
codesign --force --sign - "$TEMP_OUTPUT"
mv -f "$TEMP_OUTPUT" "$OUTPUT"
echo "✓ system-audio-capture built"
