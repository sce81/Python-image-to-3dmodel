#!/usr/bin/env bash
set -Eeuo pipefail
nvidia-smi
sudo -u ubuntu /opt/nexus/.venv-trellis2/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable after AMI provisioning"
print(torch.__version__)
print(torch.cuda.get_device_name(0))
PY
sudo -u ubuntu git -C /opt/nexus/Python-image-to-3dmodel rev-parse HEAD
sudo -u ubuntu git -C /opt/nexus/TRELLIS.2 rev-parse HEAD
