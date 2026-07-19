#!/usr/bin/env bash
set -Eeuo pipefail
: "${NEXUS_MODEL_S3_URI:?set NEXUS_MODEL_S3_URI to the private S3 prefix holding microsoft/ and facebook/ model folders}"

NEXUS_HOME=/opt/nexus
DEST="${NEXUS_HOME}/models"
aws s3 sync --only-show-errors "${NEXUS_MODEL_S3_URI%/}/" "${DEST}/"

test -f "${DEST}/microsoft/TRELLIS.2-4B/pipeline.json"
test -f "${DEST}/facebook/dinov3-vitl16-pretrain-lvd1689m/model.safetensors"
printf 'Model hydration complete: %s\n' "${DEST}"
