#!/usr/bin/env bash
set -Eeuo pipefail
: "${NEXUS_PIPELINE_GIT_URL:?missing pipeline Git URL}"
: "${NEXUS_PIPELINE_GIT_REF:?missing immutable pipeline Git ref}"
: "${NEXUS_TORCH_INDEX_URL:?missing Torch index URL}"
: "${NEXUS_TORCH_PACKAGES:?missing Torch package specification}"

NEXUS_HOME=/opt/nexus
PIPELINE_DIR="${NEXUS_HOME}/Python-image-to-3dmodel"
TRELLIS2_DIR="${NEXUS_HOME}/TRELLIS.2"
VENV_DIR="${NEXUS_HOME}/.venv-trellis2"

sudo install -d -m 0755 -o ubuntu -g ubuntu "${NEXUS_HOME}"
sudo -u ubuntu git clone "${NEXUS_PIPELINE_GIT_URL}" "${PIPELINE_DIR}"
sudo -u ubuntu git -C "${PIPELINE_DIR}" checkout --detach "${NEXUS_PIPELINE_GIT_REF}"
sudo -u ubuntu git clone https://github.com/microsoft/TRELLIS.2.git "${TRELLIS2_DIR}"
sudo -u ubuntu git -C "${TRELLIS2_DIR}" checkout --detach 75fbf0183001ed9876c8dbb35de6b68552ee08bd

sudo -u ubuntu python3.11 -m venv "${VENV_DIR}"
sudo -u ubuntu "${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel setuptools
# Use the Linux CUDA wheel index; never copy the Windows virtual environment into the AMI.
sudo -u ubuntu "${VENV_DIR}/bin/pip" install --index-url "${NEXUS_TORCH_INDEX_URL}" ${NEXUS_TORCH_PACKAGES}
if [[ -f "${TRELLIS2_DIR}/requirements.txt" ]]; then
  sudo -u ubuntu "${VENV_DIR}/bin/pip" install -r "${TRELLIS2_DIR}/requirements.txt"
fi
sudo -u ubuntu "${VENV_DIR}/bin/pip" install -e "${TRELLIS2_DIR}"
sudo -u ubuntu "${VENV_DIR}/bin/pip" install huggingface_hub boto3 awscli

sudo install -m 0755 -o ubuntu -g ubuntu "${PIPELINE_DIR}/packer/scripts/hydrate-models.sh" /usr/local/bin/nexus-trellis2-hydrate
sudo install -m 0755 -o ubuntu -g ubuntu "${PIPELINE_DIR}/packer/scripts/run-trellis2.sh" /usr/local/bin/nexus-trellis2-run
sudo install -d -m 0755 -o ubuntu -g ubuntu "${NEXUS_HOME}/models"
