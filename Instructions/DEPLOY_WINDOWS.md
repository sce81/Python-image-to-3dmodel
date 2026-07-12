# DEPLOY - Nexus Asset Pipeline MCP on Windows (RTX 4080 Super)

End-to-end, bare machine to working MCP server that your M4 Pro laptop drives over the LAN.
Follow top to bottom. Estimated time: 60-90 min, most of it model-weight download.

---

## 0. Prerequisites - what you're installing and why

| Component | Why | Version |
|---|---|---|
| NVIDIA driver + CUDA toolkit | TRELLIS geometry, native extension builds, and PBR bake | Driver 610.74 validated; Toolkit 13.3 |
| Python 3.11 | Pipeline venv (3.12 can break some 3D deps) | 3.11.x |
| Git | Clone TRELLIS | latest |
| Blender 5.1.1 | Retopo + UE-ready GLB/Blend export | 5.1.1 |
| Visual Studio Build Tools | Compiles sparse-conv / rasterizer CUDA ext | 2022, C++ workload |

You already run UE5.7 + Blender on this box, so the GPU driver and Blender are likely done.

---

## 1. System dependencies

### 1a. Verify GPU + CUDA
```powershell
nvidia-smi
```
Confirm the RTX 4080 Super shows up and the CUDA UMD reports 13.3. The validated host uses NVIDIA driver 610.74 and CUDA Toolkit 13.3.

### 1b. Install Python 3.11 (if not present)
Download from python.org, **tick "Add python.exe to PATH"** during install (your recurring PATH gotcha - do it here).
```powershell
py -3.11 --version   # should print 3.11.x
```

### 1c. Install Git + Blender (skip if present)
```powershell
winget install --id Git.Git -e
winget install --id BlenderFoundation.Blender -e
```
Note Blender's install path - the server expects `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`. Adjust `BLENDER` in `mcp_server.py` if yours differs.

### 1d. Visual Studio Build Tools (for CUDA extensions)
```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```
In the installer, select **"Desktop development with C++"**. Required to compile TRELLIS's sparse-convolution and rasterizer kernels.

---

## 2. Project root

Work from the existing repository:

```powershell
cd C:\Users\simon\Documents\3D-Imaging-Pipeline
```

The setup script expects the checked-in .tmp extension sources and the external TRELLIS checkout documented below.

---

## 3. Python environment

```powershell
.\setup_cuda13.ps1
.\.venv\Scripts\Activate.ps1
```
Sanity check CUDA is visible to torch:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA GeForce RTX 4080 SUPER
```
If that prints False, rerun setup_cuda13.ps1 and confirm CUDA Toolkit 13.3 is installed at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3.

---

## 4. Install TRELLIS 2

```powershell
# still in the venv
cd C:\Users\simon\Documents\UnrealEngine-ProjectWork
git clone https://github.com/microsoft/TRELLIS.git
cd TRELLIS
git checkout 442aa1e1afb9014e80681d3bf604e8d728a86ee7
git submodule update --init --recursive
```
The pipeline loads this checkout through TRELLIS_DIR in generate_props.py. setup_cuda13.ps1 installs the tested Python stack, keeps the cu124 spconv runtime side-by-side, and rebuilds nvdiffrast plus the Gaussian rasterizer against CUDA Toolkit 13.3.

**First run downloads weights (~15 GB)** from Hugging Face. Pre-approve the gated models (DINOv3, RMBG-2.0) on huggingface.co, then:
```powershell
hf auth login
```
Now edit `generate_props.py`: the TRELLIS import paths and `pipe.run()` signature at the top are written to the standard repo layout - confirm the module names match your cloned version (this is the one place versions drift).

---

## 5. Smoke-test the pipeline locally (before MCP)

Drop one test PNG into `Inputs\`, then:
```powershell
cd C:\Users\simon\Documents\3D-Imaging-Pipeline
.\.venv\Scripts\Activate.ps1
python generate_props.py          # -> raw_meshes\*.glb
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python blender_postprocess.py   # -> Outputs\*.glb
```
If you get a GLB in `Outputs\`, the pipeline works. Fix this before touching MCP - the server just wraps these two commands.

---

## 6. Configure + launch the MCP server

### 6a. Set the shared secret
```powershell
setx NEXUS_MCP_TOKEN "paste-a-long-random-string-here"
```
Close and reopen the terminal so the env var loads. Set the **same** string on the laptop later.

### 6b. Open the firewall port (Private networks only)
Run PowerShell **as Administrator**:
```powershell
New-NetFirewallRule -DisplayName "Nexus MCP" -Direction Inbound -LocalPort 8765 -Protocol TCP -Action Allow -Profile Private
```

### 6c. Find this machine's LAN IP
```powershell
ipconfig
# note the IPv4 Address, e.g. 192.168.1.42
```

### 6d. Launch
```powershell
cd C:\Users\simon\Documents\3D-Imaging-Pipeline
.\.venv\Scripts\Activate.ps1
python mcp_server.py
# -> serving at http://0.0.0.0:8765/mcp
```
Leave this terminal running. That's the server.

---

## 7. Connect from the M4 Pro laptop

Add to your MCP client config (Claude Desktop / Claude Code):
```json
{
  "mcpServers": {
    "nexus-assets": {
      "type": "http",
      "url": "http://192.168.1.42:8765/mcp"
    }
  }
}
```
Use your actual Windows IP. Restart the client. You should see the four tools:
`generate_prop`, `postprocess`, `get_mesh`, `list_meshes`. Pass `token` = your NEXUS_MCP_TOKEN on each call.

Test from the laptop: ask it to call `list_meshes` - if it returns your smoke-test mesh, the full loop works.

---

## 8. Run it as a background service (optional but recommended)

So you don't need a terminal open. Easiest reliable option is **NSSM**:
```powershell
winget install --id NSSM.NSSM -e
nssm install NexusMCP
```
In the GUI: set **Application** to `C:\Users\simon\Documents\3D-Imaging-Pipeline\.venv\Scripts\python.exe`, **Arguments** to `mcp_server.py`, and **Startup directory** to `C:\Users\simon\Documents\3D-Imaging-Pipeline`.
```powershell
nssm start NexusMCP
```
Now it survives reboots and runs headless.

---

## 9. Remote access beyond the LAN (optional)

To drive it from a client site or caf-, do NOT port-forward 8765. Install Tailscale on both machines:
```powershell
winget install --id Tailscale.Tailscale -e
```
Sign in on both, then point the laptop's MCP config at the Windows machine's **Tailscale IP** (100.x.x.x) instead of the LAN IP. Same port, private encrypted network, nothing exposed publicly.

---

## Troubleshooting quick table

| Symptom | Cause | Fix |
|---|---|---|
| torch.cuda.is_available() False | driver/Torch mismatch | rerun setup_cuda13.ps1 and verify driver 610.74 + Toolkit 13.3 |
| TRELLIS build fails | no C++ Build Tools | install step 1d, reopen terminal |
| Server starts, tool calls hang | Blender path wrong | fix `BLENDER` in mcp_server.py |
| Laptop can't reach server | firewall / wrong IP | recheck step 6b/6c; same subnet |
| `PermissionError: Bad token` | token mismatch | same NEXUS_MCP_TOKEN both sides |
| Mesh generates, no texture | using Mac fallback | texturing is Windows/CUDA only |

---

## What "done" looks like
From the laptop you say: *"Generate a prop from this reference, call it corp-terminal-01, then postprocess it at 6000 tris and send me the GLB."* The 4080 generates + textures, Blender retopos, and the finished UE-ready GLB comes back to the Mac - zero per-asset cost, full topology control, no Meshy.
