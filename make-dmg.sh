#!/bin/bash
# Build MyTranslator.dmg — drag-and-drop installer (matches Hen's install UX)
set -euo pipefail
cd "$(dirname "$0")"

# 1) build the .app first unless a verified bundle was just produced.
if [[ "${MYT_SKIP_APP_BUILD:-0}" != "1" ]]; then
  ./make-app.sh
fi

# 2) stage a folder with app + Applications symlink
STAGE="/tmp/myt-dmg-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/.background"
trap 'rm -rf "$STAGE"' EXIT
cp -R MyTranslator.app "$STAGE/"
ln -s /Applications "$STAGE/Applications"

# 3) simple background PNG (dark, 660x400)
python3 - << 'PY'
from PIL import Image, ImageDraw
img = Image.new('RGB', (660, 400), (28, 28, 30))
d = ImageDraw.Draw(img)
d.rounded_rectangle([30, 90, 300, 320], radius=20, fill=(44, 44, 46), outline=(80,80,84), width=2)
d.rounded_rectangle([360, 90, 630, 320], radius=20, fill=(44, 44, 46), outline=(80,80,84), width=2)
d.text((105, 185), "My Translator", fill=(235,235,245))
d.text((455, 185), "拖到此处", fill=(235,235,245))
img.save('/tmp/myt-dmg-stage/.background/bg.png')
PY

# 4) create the DMG with a nice layout
hdiutil create -volname "My Translator" \
  -srcfolder "$STAGE" \
  -ov -format UDZO \
  MyTranslator.dmg

echo "✓ MyTranslator.dmg 构建完成"
echo "  双击挂载 → 拖拽 MyTranslator 到 Applications 即完成安装"
ls -lh MyTranslator.dmg | awk '{print "  大小:", $5}'
