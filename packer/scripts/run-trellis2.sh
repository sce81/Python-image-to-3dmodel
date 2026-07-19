#!/usr/bin/env bash
set -Eeuo pipefail
NEXUS_HOME=/opt/nexus
: "${NEXUS_SINGLE:?set NEXUS_SINGLE to an approved geometry raster path}"
: "${NEXUS_ASSET_NAME:?set NEXUS_ASSET_NAME}"

export NEXUS_TRELLIS2_DIR="${NEXUS_HOME}/TRELLIS.2"
export NEXUS_T2_MODEL="${NEXUS_HOME}/models/microsoft/TRELLIS.2-4B"
export NEXUS_DINOV3_PATH="${NEXUS_HOME}/models/facebook/dinov3-vitl16-pretrain-lvd1689m"

# G7.2xlarge has 32 GB VRAM: use the full official 4B checkpoint and its
# highest-resolution cascade. Set NEXUS_T2_PIPELINE_TYPE=1024_cascade only
# when a faster, lower-resolution experiment is intentional.
export NEXUS_T2_PIPELINE_TYPE="${NEXUS_T2_PIPELINE_TYPE:-1536_cascade}"
exec "${NEXUS_HOME}/.venv-trellis2/bin/python" "${NEXUS_HOME}/Python-image-to-3dmodel/generate_props_t2.py"
