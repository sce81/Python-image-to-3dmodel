param(
    [string]$Venv = ".venv"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$Nvcc = Join-Path $CudaHome "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $Nvcc)) {
    throw "CUDA Toolkit 13.3 is required at $CudaHome"
}

py -3.11 -m venv $Venv
$Python = Join-Path $Root "$Venv\Scripts\python.exe"

& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu132
& $Python -m pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130 --no-deps
& $Python -m pip install spconv-cu124==2.3.8 nvidia-cuda-runtime-cu12==12.4.127 nvidia-cuda-nvrtc-cu12==12.4.127

$env:CUDA_HOME = $CudaHome
$env:CUDA_PATH = $CudaHome
$env:PATH = "$CudaHome\bin;$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = "8.9"

$BuildDirs = @(
    ".tmp\nvdiffrast\build",
    ".tmp\mip-splatting\submodules\diff-gaussian-rasterization\build"
)
foreach ($BuildDir in $BuildDirs) {
    $BuildPath = Join-Path $Root $BuildDir
    if (Test-Path -LiteralPath $BuildPath) {
        Remove-Item -LiteralPath $BuildPath -Recurse -Force
    }
}

& $Python -m pip install --no-build-isolation --no-cache-dir -r .\requirements-cu132-lock.txt --extra-index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://download.pytorch.org/whl/cu130

& $Python -m pip check
& $Python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"