#!/usr/bin/env bash
set -e

MANIFEST=third_party/third_party_weights_manifest.json
if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

# Check git lfs
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git lfs not found. Installing..."
  git lfs install
fi

echo "Preparing .gitattributes via git lfs track for entries in $MANIFEST"
python - <<'PY'
import json,os,subprocess
m=json.load(open('third_party/third_party_weights_manifest.json'))
paths=set()
for e in m:
    p=e['path']
    # Convert to repo-root relative path (ensure leading ./ removed)
    paths.add(p)
for p in sorted(paths):
    print('Tracking',p)
    subprocess.run(['git','lfs','track',p], check=True)
print('\nDone. Review .gitattributes, then add and commit it:\n  git add .gitattributes\n  git commit -m "chore: track large weights with git lfs"')
PY

echo "Note: After committing .gitattributes, add the large weight files (if present) and push. Be aware of Git LFS quotas on your Git host."