#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  ubuntu-drivers-common linux-headers-"$(uname -r)" build-essential \
  ca-certificates curl git git-lfs awscli jq unzip \
  python3.11 python3.11-venv python3-pip

# Select Ubuntu's supported datacenter/GPU driver for the actual G7 hardware.
sudo ubuntu-drivers install --gpgpu
sudo systemctl enable amazon-ssm-agent || true
