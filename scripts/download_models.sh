#!/usr/bin/env bash
# Download the AI-Glasses YOLO + MediaPipe models from ModelScope and verify
# their SHA256 against the authoritative values fetched from ModelScope's
# repo-files API. Skips files already present with a matching hash.
#
# Usage:
#   ./scripts/download_models.sh           # download into ./model/
#   MODEL_DIR=/path/to/dir ./scripts/download_models.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./model}"
BASE_URL="https://www.modelscope.cn/models/archifancy/AIGlasses_for_navigation/resolve/master"

# name<TAB>sha256<TAB>size — canonical values from the ModelScope files API
FILES=(
  "hand_landmarker.task	fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1	7819105"
  "shoppingbest5.pt	77625f14054db17a4ef3eae033a026e610e020bf0d1c34502c81795460b3b5cc	144020552"
  "trafficlight.pt	4a834b25ad6977f8dd9b80e01f81c558c7a1f56028f9a2c57dfe6164e68ab5a0	175069538"
  "yolo-seg.pt	5dfed5a1bc47bf609577c45ac374bd5e0bcab62d1f04d1fd02392cf0af3b9abc	144014984"
  "yoloe-11l-seg.pt	a993fb0fc7c8830939ae14e6434a925dd1179428158c2761482eb8a8d8a3699f	70982416"
)

if command -v sha256sum >/dev/null 2>&1; then
  SHA256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  SHA256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  echo "error: need sha256sum or shasum on PATH" >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

for entry in "${FILES[@]}"; do
  name="${entry%%	*}"
  rest="${entry#*	}"
  expected_sha="${rest%%	*}"
  expected_size="${rest##*	}"

  if [[ -f "$name" ]]; then
    actual_sha=$(SHA256 "$name")
    if [[ "$actual_sha" == "$expected_sha" ]]; then
      echo "[OK ] $name (already present, sha256 match)"
      continue
    fi
    echo "[!! ] $name exists but sha256 mismatch — redownloading"
    rm -f "$name"
  fi

  echo "[GET] $name ($(printf '%.1f' $(echo "$expected_size / 1048576" | bc -l)) MiB) ..."
  tmp="$name.partial"
  curl -fL --retry 3 -o "$tmp" "$BASE_URL/$name"

  actual_sha=$(SHA256 "$tmp")
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "  expected sha256: $expected_sha" >&2
    echo "  got      sha256: $actual_sha"   >&2
    echo "[FAIL] $name — sha256 mismatch, refusing to use" >&2
    rm -f "$tmp"
    exit 1
  fi
  mv "$tmp" "$name"
  echo "[OK ] $name (downloaded, sha256 verified)"
done

echo ""
echo "All models present in $(pwd) and verified against ModelScope canonical hashes."
