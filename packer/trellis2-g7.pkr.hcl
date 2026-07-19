packer {
  required_plugins {
    amazon = {
      version = "= 1.8.1"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "aws_region" {
  type        = string
  description = "AWS Region for the AMI build and resulting AMI."
}

variable "subnet_id" {
  type        = string
  description = "Private build subnet with NAT egress and SSM connectivity."
}

variable "iam_instance_profile" {
  type        = string
  description = "Build-instance profile with AmazonSSMManagedInstanceCore and read-only access to the approved model S3 prefix."
}

variable "pipeline_git_url" {
  type        = string
  description = "Git URL containing this pipeline."
  default     = "https://github.com/sce81/Python-image-to-3dmodel.git"
}

variable "pipeline_git_ref" {
  type        = string
  description = "Immutable pushed commit SHA or signed tag to bake into the AMI."
}

variable "ami_name_prefix" {
  type    = string
  default = "nexus-trellis2-g7"
}

variable "root_volume_gb" {
  type    = number
  default = 200
}

variable "torch_index_url" {
  type    = string
  default = "https://download.pytorch.org/whl/cu130"
}

variable "torch_packages" {
  type    = string
  default = "torch==2.10.0 torchvision"
}

locals {
  ami_name = "${var.ami_name_prefix}-${formatdate("YYYYMMDD-hhmmss", timestamp())}"
}

source "amazon-ebs" "trellis2_g7" {
  region                      = var.aws_region
  spot_instance_types         = ["g7.2xlarge"]
  spot_allocation_strategy    = "price-capacity-optimized"
  ami_name                    = local.ami_name
  ami_description             = "TRELLIS.2 worker runtime; model weights hydrate from private S3 at launch."
  ssh_username                = "ubuntu"
  communicator                = "ssh"
  ssh_interface               = "session_manager"
  iam_instance_profile        = var.iam_instance_profile
  subnet_id                   = var.subnet_id
  associate_public_ip_address = false
  encrypt_boot                = true

  source_ami_filter {
    filters = {
      name                = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["099720109477"]
  }

  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  run_tags = {
    Name      = "${local.ami_name}-builder"
    Component = "trellis2"
    ManagedBy = "packer"
  }

  tags = {
    Name                = local.ami_name
    Component           = "trellis2"
    PipelineGitRef      = var.pipeline_git_ref
    ModelHydration      = "required"
    ModelProfile        = "TRELLIS.2-4B-full-1536-cascade"
    PublicIngress       = "disabled"
    ManagedBy           = "packer"
  }
}

build {
  name    = "trellis2-g7-ami"
  sources = ["source.amazon-ebs.trellis2_g7"]

  provisioner "shell" {
    script = "packer/scripts/01-install-nvidia.sh"
  }

  provisioner "shell" {
    inline            = ["sudo reboot"]
    expect_disconnect = true
  }

  provisioner "shell" {
    pause_before = "20s"
    environment_vars = [
      "NEXUS_PIPELINE_GIT_URL=${var.pipeline_git_url}",
      "NEXUS_PIPELINE_GIT_REF=${var.pipeline_git_ref}",
      "NEXUS_TORCH_INDEX_URL=${var.torch_index_url}",
      "NEXUS_TORCH_PACKAGES=${var.torch_packages}",
    ]
    script = "packer/scripts/02-install-trellis2.sh"
  }

  provisioner "shell" {
    script = "packer/scripts/03-verify-image.sh"
  }

  post-processor "manifest" {
    output     = "packer/manifest.json"
    strip_path = true
    custom_data = {
      pipeline_git_ref = var.pipeline_git_ref
      instance_type    = "g7.2xlarge"
      model_hydration  = "private-s3-at-launch"
      model_profile    = "TRELLIS.2-4B-full-1536-cascade"
    }
  }
}
