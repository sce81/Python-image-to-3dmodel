# TRELLIS.2 G7 Packer AMI

This creates a private Ubuntu 24.04 AMI on a Spot `g7.2xlarge`. It installs Python 3.11, the supported NVIDIA datacenter driver, the pinned TRELLIS.2 checkout, the isolated T2 virtual environment, and the Nexus pipeline at a supplied immutable Git revision.

The G7 runtime profile is the full official **`microsoft/TRELLIS.2-4B`** checkpoint with **`1536_cascade`** as its default: this is the 32 GB VRAM quality profile. It does not use the ComfyUI FP8 model. Set `NEXUS_T2_PIPELINE_TYPE=1024_cascade` only for a deliberate faster/lower-resolution run.

It deliberately does **not** bake model weights or credentials. `microsoft/TRELLIS.2-4B` and gated DINOv3 weights must be license-approved and copied to a private S3 prefix before launch. At runtime, an instance role with read-only access to that prefix runs `nexus-trellis2-hydrate`.

## Prerequisites

- Packer and AWS CLI on the build machine; authenticated AWS credentials with AMI, snapshot, EC2 Fleet/Spot, subnet, IAM-profile pass, and SSM build permissions.
- A private subnet with NAT egress and SSM connectivity. Do not assign a public IP or open MCP, ComfyUI, or SSH.
- Build instance profile with `AmazonSSMManagedInstanceCore` and minimal read-only S3 access to the approved model prefix.
- A pushed, immutable Git commit SHA for `pipeline_git_ref`.
- Accepted Hugging Face licenses and a private S3 copy of the model folders. Never pass an HF token to Packer or store one in a vars file.

## Build

```powershell
Copy-Item .\packer\variables.pkrvars.hcl.example .\packer\trellis2-g7.auto.pkrvars.hcl
# Edit only the untracked .auto.pkrvars.hcl file.
packer init .\packer
packer fmt -check .\packer
packer validate -var-file=.\packer\trellis2-g7.auto.pkrvars.hcl .\packer
packer build -var-file=.\packer\trellis2-g7.auto.pkrvars.hcl .\packer
```

The Packer Amazon builder uses `price-capacity-optimized` for the G7 Spot build rather than `lowest-price`; this is the lowest resilient cost path, not a promise of one fixed region or hourly rate. The build may still be interrupted and should be retried from the template.

## Launch and hydrate

After launching the AMI with a private instance role:

```bash
export NEXUS_MODEL_S3_URI=s3://approved-private-bucket/trellis2-models
nexus-trellis2-hydrate
export NEXUS_SINGLE=/data/approved_geometry.png
export NEXUS_ASSET_NAME=asset_id
export NEXUS_GEOMETRY_ONLY=1
# Defaults to the G7 32 GB quality profile: TRELLIS.2-4B + 1536_cascade.
# Override only for an intentional faster experiment:
# export NEXUS_T2_PIPELINE_TYPE=1024_cascade
nexus-trellis2-run
```

Sync source rasters, logs, manifests, checkpoints, and reviewed outputs to private S3 between atomic stages. Treat the instance as disposable and terminate it with its transient EBS volume after results are recovered.
