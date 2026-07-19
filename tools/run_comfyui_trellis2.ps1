# Launch ComfyUI with the TRELLIS.2 stack (isolated venv, Torch 2.10+cu130).
# The main pipeline venv (Torch 2.13/cu13.2, TRELLIS 1) is untouched by this.
#
#   .\tools\run_comfyui_trellis2.ps1              # GUI + API on 127.0.0.1:8188
#   .\tools\run_comfyui_trellis2.ps1 -Port 8199
#
# First TRELLIS.2 generation notes:
# - In the Trellis2LoadModel node: use modelname visualbruno/TRELLIS.2-4B-FP8
#   (16 GB card), backend sdpa, low_vram true, conv_backend flex_gemm.
# - DINOv3 must exist at models\facebook\dinov3-vitl16-pretrain-lvd1689m\
#   (gated; see AGENTS.md TRELLIS.2 section).

param(
    [int]$Port = 8188
)

$comfy = Join-Path $PSScriptRoot "ComfyUI"
$python = Join-Path $comfy ".venv-trellis2\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "TRELLIS.2 venv not found: $python"
    exit 1
}
# 16 GB card: reduce fragmentation, keep ~0.5 GB for the desktop compositor,
# and release model VRAM between workflow stages instead of caching it.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
Set-Location $comfy
& $python main.py --port $Port --listen 127.0.0.1 --reserve-vram 0.5 --disable-smart-memory
