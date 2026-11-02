#!/bin/bash
# Build MyTranslator.app — the .app wrapper is REQUIRED for macOS TCC
# to show the microphone permission prompt.
set -euo pipefail
cd "$(dirname "$0")"

APP="MyTranslator.app"
BUILD_APP=".MyTranslator.app.build"
OLD_APP=".MyTranslator.app.previous"
SIGN_IDENTITY="${MYT_CODESIGN_IDENTITY:-}"
if [ -z "$SIGN_IDENTITY" ]; then
  SIGN_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | sed -n 's/.*"\(Apple Development: [^"]*\)".*/\1/p' \
    | head -n 1)"
fi
rm -rf "$BUILD_APP"
mkdir -p "$BUILD_APP/Contents/MacOS" "$BUILD_APP/Contents/Resources/bin"
trap 'rm -rf "$BUILD_APP"' EXIT

cat > "$BUILD_APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>My Translator</string>
    <key>CFBundleDisplayName</key>
    <string>My Translator</string>
    <key>CFBundleIdentifier</key>
    <string>com.rain.mytranslator</string>
    <key>CFBundleVersion</key>
    <string>0.3.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.3.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>my-translator</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>My Translator 需要访问麦克风以识别语音并进行实时翻译。</string>
    <key>NSAudioCaptureUsageDescription</key>
    <string>My Translator 需要捕获系统音频，以直接翻译视频、会议和播放器中的声音。</string>
    <key>NSSpeechRecognitionUsageDescription</key>
    <string>My Translator 使用本地语音识别将你说的话翻译成目标语言。</string>
    <key>LSEnvironment</key>
    <dict>
        <key>MYT_AUTO_VOICE</key>
        <string>1</string>
    </dict>
</dict>
</plist>
PLIST

# Release builds materially reduce backend overhead and make the application
# portable. Refuse to build if a model is missing, modified, or linked from an
# external application.
./tools/verify_models.sh
./tools/build-system-audio.sh
./tools/build-qwen-asr-worker.sh
./tools/build-qwen-translation-worker.sh
cargo build --release --locked
cp target/release/my-translator "$BUILD_APP/Contents/MacOS/"
cp -R bin/llama-dist "$BUILD_APP/Contents/Resources/bin/"
cp -R bin/whisper-dist "$BUILD_APP/Contents/Resources/bin/"
cp -R bin/whisper-server-dist "$BUILD_APP/Contents/Resources/bin/"
cp -R bin/vad-dist "$BUILD_APP/Contents/Resources/bin/"
cp -R bin/qwen-asr-dist "$BUILD_APP/Contents/Resources/bin/"
cp -R bin/qwen-translation-dist "$BUILD_APP/Contents/Resources/bin/"
cp bin/system-audio-capture "$BUILD_APP/Contents/Resources/bin/"
cp THIRD_PARTY_NOTICES.md "$BUILD_APP/Contents/Resources/"
# Models are project-owned regular files validated above.
cp -R models "$BUILD_APP/Contents/Resources/"
if [ -n "$SIGN_IDENTITY" ]; then
  codesign --force --deep --timestamp=none --sign "$SIGN_IDENTITY" "$BUILD_APP"
  SIGNING_LABEL="Apple Development ($SIGN_IDENTITY)"
else
  codesign --force --deep --sign - "$BUILD_APP"
  SIGNING_LABEL="adhoc（升级后可能需要重新授权系统音频）"
fi

# Publish the fully built bundle in one rename. The live process never sees a
# half-copied model directory if this script runs while MyTranslator is open.
rm -rf "$OLD_APP"
if [ -e "$APP" ]; then
  mv "$APP" "$OLD_APP"
fi
if ! mv "$BUILD_APP" "$APP"; then
  if [ -e "$OLD_APP" ]; then
    mv "$OLD_APP" "$APP"
  fi
  exit 1
fi
rm -rf "$OLD_APP"

echo "✓ $APP 构建完成（${SIGNING_LABEL}）"
du -sh "$APP" | awk '{print "  完整大小:", $1}'
echo "  启动: open $APP --args --voice"
