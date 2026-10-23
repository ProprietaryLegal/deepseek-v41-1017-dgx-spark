#!/usr/bin/env bash
# stage_overlay.sh NODE [NODE...]: production staging of the remote-DSpark overlay (configuration D). Copies into
# NODE:~/dsv41-offload-prod/dspark/ the remote_dspark package, the spec_decode __init__.py overlay and the ofl transport
# (libofl.so + ofl.py), verifies every copy, and checks that the image's own spec_decode/__init__.py is the file the
# overlay was made from. dsv41.env's OFFLOAD_DOCKER_ARGS bind-mounts these paths read-only into the serve container.
# Coordinator-only. Run from the repo root on the head node (start.sh does it when OFFLOAD_DSPARK_ENV is set):
#   offload/dspark/spark/stage_overlay.sh spark spark-b
# Remove:  ssh NODE rm -rf ~/dsv41-offload-prod/dspark   (nothing reads it once OFFLOAD_DSPARK_ENV is unset)
set -euo pipefail
cd "$(dirname "$0")/../../.."
PKG=offload/dspark/spark/remote_dspark
INIT=offload/dspark/spark/overlay/vllm/v1/worker/gpu/spec_decode/__init__.py
OFL_PY=offload/transport/ofl.py
# libofl.so is an aarch64 build product (offload/dspark/INTEGRATION.md 1.3); the staged copy must be this exact build
OFL_SO_SRC=${OFL_SO_SRC:-$HOME/dsv41-offload-run/ofl/libofl.so}
OFL_SO_SHA=4daef6c28f4d777817d2bc8e2af1a9053f2dea7e54bbf8871f97c762bcf8d8e5
IMAGE_INIT_SHA=61b3acba401c1079039496f72445c28bbc6119d0b508f9a75d231f305aa1b7e8
IMAGE=${IMAGE:-ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5}
IMG_INIT=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/__init__.py
DEST=dsv41-offload-prod/dspark
[ "$(sha256sum < "$OFL_SO_SRC" | cut -d' ' -f1)" = "$OFL_SO_SHA" ] \
  || { echo "$OFL_SO_SRC is not the pinned libofl.so build ($OFL_SO_SHA)" >&2; exit 1; }
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/remote_dspark" "$stage/ofl"
(cd "$PKG" && find . -name '*.py' -not -path '*/__pycache__/*' -print0 | cpio -0pdm --quiet "$stage/remote_dspark")
cp "$INIT" "$stage/spec_decode__init__.py"
cp "$OFL_PY" "$OFL_SO_SRC" "$stage/ofl/"
local_sums=$(cd "$stage" && find . -type f | LC_ALL=C sort | xargs sha256sum)
for n in "$@"; do
  echo "== $n"
  ssh -o BatchMode=yes "$n" "rm -rf \$HOME/$DEST && mkdir -p \$HOME/$DEST"
  tar cf - -C "$stage" . | ssh -o BatchMode=yes "$n" "tar xf - -C \$HOME/$DEST"
  remote_sums=$(ssh -o BatchMode=yes "$n" "cd \$HOME/$DEST && find . -type f | LC_ALL=C sort | xargs sha256sum")
  [ "$local_sums" = "$remote_sums" ] || { echo "$n: staged copies differ" >&2; exit 1; }
  echo "  staged $(printf '%s\n' "$local_sums" | wc -l) files, sha256 match"
  img=$(ssh -o BatchMode=yes "$n" "docker run --rm --entrypoint sha256sum $IMAGE $IMG_INIT" | cut -d' ' -f1)
  [ "$img" = "$IMAGE_INIT_SHA" ] || { echo "$n: the image's spec_decode/__init__.py ($img) is not the overlay's base" >&2; exit 1; }
  echo "  image spec_decode/__init__.py matches the overlay's base"
done
