#!/usr/bin/env bash
set -e

# 自动将本地原始权重复制或软链接到 third_party 目录的占位位置
# 如果你的权重在原始子仓库位置（如 croco/pretrained_models/...），这个脚本会将它们复制到 third_party

MANIFEST="third_party/third_party_weights_manifest.json"
if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

python - <<'PY'
import json,os,shutil
m=json.load(open('third_party/third_party_weights_manifest.json'))
for e in m:
    src=e['path']
    dest=os.path.join('third_party', e['repo'], os.path.relpath(src, e['repo']))
    dest_dir=os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(src):
        print('Copying', src, '->', dest)
        shutil.copy2(src, dest)
    else:
        print('MISSING local file:', src)
        print('Please download it to:', src, '\nor place it and re-run this script')
PY

echo "Done. If files were missing, place them at the original paths listed in the manifest and re-run this script."