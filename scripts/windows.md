# Windows setup notes

This doc is the catalog of things that go wrong on a fresh Windows box and how to fix them. The happy path:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    # one-time, allows .ps1 to run
$env:BONSAI_TOKEN = 'hf_...'                           # until weights go public
.\setup.ps1                                            # installs uv, torch+cu128, triton-windows, gemlite, hqq, backend_gpu, model
.\scripts\serve.ps1                                    # backend on :8000 + Next.js studio on :3000
```

Everything below is for when that isn't enough.

Tested on Windows 11 + RTX 3060 Laptop + driver 566.07 + Python 3.11.

## Required before running setup.ps1

Nothing else in the demo installs these for you. Check each one if setup misbehaves.

| What | Why it matters | Verify | Get it |
|---|---|---|---|
| Windows 10/11 x64 | triton-windows ships x64 wheels only | `[Environment]::Is64BitOperatingSystem` returns `True` | n/a |
| NVIDIA driver 566.07+ | torch cu128 wheels need a CUDA 12.8-capable driver. Older drivers run torch in CPU mode silently | `nvidia-smi` shows driver `566.07` or higher | https://www.nvidia.com/Download/index.aspx |
| Git for Windows | setup.ps1 clones `vendor/image-studio` and `vendor/mflux-prism` | `git --version` works in PowerShell | https://git-scm.com/download/win |
| Execution policy RemoteSigned (CurrentUser) | otherwise PowerShell refuses to run any `.ps1` | `Get-ExecutionPolicy -Scope CurrentUser` returns `RemoteSigned` or `Unrestricted` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or invoke each script with `powershell -ExecutionPolicy Bypass -File .\setup.ps1` |
| Visual C++ Runtime 14.42+ | triton-windows loads a native DLL (`libtriton`). Missing this gives `ImportError: DLL load failed`. Most dev machines already have it from VS, Python installers, or another tool | run setup; if triton import works in the post-install smoke test, you're fine | https://aka.ms/vs/17/release/vc_redist.x64.exe |
| ~15 GB free disk | torch wheel 2.6 GB, model 4 GB, node_modules ~700 MB, plus Triton/gemlite caches that grow over time | `Get-PSDrive C` | free space |
| HuggingFace token with access to `prism-ml/bonsai-image-*` | weights are gated until the public launch | login at https://huggingface.co | set `$env:BONSAI_TOKEN = 'hf_...'` before running setup.ps1 |

System Python is **not** required. uv will fetch its own Python 3.11.

## FAQ: failure modes and fixes

### setup.ps1: "running scripts is disabled on this system"

PowerShell's default execution policy is `Restricted`. Fix it once for your account, then every `.ps1` in this repo runs normally:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If you can't change the policy (locked-down corporate machine), bypass per-invocation instead:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\serve.ps1
```

### setup.ps1: "'git' is not recognized"

Git for Windows isn't installed or isn't on PATH. Install from https://git-scm.com/download/win, open a fresh PowerShell (so the new PATH is picked up), then re-run setup.

### setup.ps1: "torch+cu128 install failed" / pip resolver error

Two common causes:

1. **Old NVIDIA driver.** torch 2.11 + cu128 needs driver 566.07 or newer. Run `nvidia-smi`. If the driver is older, update from https://www.nvidia.com/Download/index.aspx and reboot.
2. **uv version pinned a torch the index doesn't carry.** The script pins `torch==2.11.*`. If PyTorch later moves on, you may need to bump this (and `triton-windows` along with it, per the compat matrix in [setup.ps1](../setup.ps1)).

### setup.ps1: triton-windows installed but `import triton` fails with "DLL load failed"

Install Visual C++ Runtime 14.42+ from https://aka.ms/vs/17/release/vc_redist.x64.exe and re-run the import test:

```powershell
.\.venv\Scripts\python.exe -c "import triton; print(triton.__version__)"
```

### serve.ps1: backend boots but `/generate` returns 500 "Gemlite transformer artifact not found at ..."

`backend_gpu` keeps separate transformer paths per arm (`MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH` for ternary, `MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH` for binary). When a `/generate` request targets an arm whose weights aren't on disk, the loader raises FileNotFoundError.

Three flavors of this error:

1. **`...artifact not found at \root\models\bonsai-binary`** — old failure mode. Means serve.ps1 didn't wire the env vars at all. Fix: re-run from `main`; serve.ps1 now sets both `*_TRANSFORMER_PATH` vars to whatever variants are on disk and prints a `Transformer pool:` line at startup so you can see which arms are wired.

2. **`...artifact not found at C:\...\models\bonsai-image-4B-binary-gemlite\transformer-gemlite-*`** — env vars are correct, but you haven't downloaded the binary variant. Either:
   - Download it: `.\scripts\download_model.ps1 binary`.
   - Or send requests with `"backend": "bonsai-ternary-gemlite"` (which `send_request.ps1` does automatically by probing `/backends`).

3. **Frontend studio dropdown lets you pick binary, but the request fails** — same as #2 above. The dropdown lists all *known* arms, not all *downloaded* arms. Download the missing variant or stick to the loaded one.

If you flip `BONSAI_VARIANT=binary` and re-run serve, the binary arm becomes the boot-time default. Both variants stay wired as long as their weight dirs exist under `models/`.

### serve.ps1: "Port 8000 already in use" / "Port 3000 already in use"

Something else is bound to those ports. Either kill it or move the demo:

```powershell
# find who owns 8000
Get-NetTCPConnection -LocalPort 8000 -State Listen | Format-Table OwningProcess, State
Get-Process -Id <pid>

# or just use different ports
$env:BACKEND_PORT = '8800'
$env:FRONTEND_PORT = '3100'
.\scripts\serve.ps1
```

### generate.py crashes with `UnicodeEncodeError: 'charmap' codec can't encode character '⏳'`

You ran `python scripts\generate.py` directly. Windows defaults stdout to cp1252, which can't encode the banner glyphs. Use the wrapper `.\scripts\generate.ps1` (which sets `PYTHONUTF8=1`), or set it yourself before running:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe .\scripts\generate.py --force-gpu-run -p "..."
```

### generate.ps1 / serve.ps1: CUDA out of memory at larger resolutions

The 3060 Laptop has 6 GB VRAM. 512x512 needs ~5 GB, so 1024x1024 will OOM. Stick to fast presets (512x512, 624x416, 416x624). On a 12 GB+ card the full quality presets work.

### download_model.ps1: "401 Client Error" / "Repository Not Found"

Either the token isn't set, the token doesn't have access to `prism-ml/bonsai-image-*`, or there's a typo:

```powershell
$env:BONSAI_TOKEN = 'hf_yourtokenhere'
.\scripts\download_model.ps1 ternary
```

If you've already run `huggingface-cli login` (or setup.ps1 ran with `BONSAI_TOKEN` set), the token is cached in `%USERPROFILE%\.cache\huggingface\token` and the env var is optional.

### uv sync or pip install hangs / fails partway

- Antivirus is sometimes the culprit. It can quarantine `.pyd` files mid-install. Whitelist `.venv\`.
- Stale cache: `uv cache clean` (or just delete `.venv\` and re-run setup).
- Corporate proxy: set `$env:HTTPS_PROXY` and `$env:HTTP_PROXY` before running setup.

### "Triton: torch.compile cache write failed" or weird path-too-long errors

Windows' default 260-char path limit (MAX_PATH) bites the Triton kernel cache. Two fixes:

- One-time: enable long paths in registry. In elevated PowerShell:
  ```powershell
  New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force
  ```
  Reboot.
- Or, move the repo closer to the drive root (e.g. `C:\Bonsai-Image-Demo` instead of `C:\Users\you\Desktop\workspace\Bonsai-Image-Demo`).

### Slow first inference, fast subsequent ones

Expected. First call at any new resolution pays:

- ~30s of Triton kernel JIT compile for that shape
- ~10s of gemlite autotune search

Both are cached under `outputs\.triton_cache\` and `outputs\.gemlite_cache\autotune.json`. Subsequent calls at the same shape skip both. Delete those dirs only if you suspect a stale cache is causing crashes.

### "Setup complete" but `torch.cuda.is_available()` is False

Driver problem, not a Python one. Check:

```powershell
nvidia-smi                              # must show your GPU + driver >= 566.07
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If `nvidia-smi` works but `torch.cuda.is_available()` is False, the wrong torch wheel got installed (probably the CPU-only one). Re-run with an explicit reinstall:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
uv pip install --python .\.venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.*'
```

### Resetting from scratch

**Don't delete `models/`** unless the weights themselves are actually broken. The model files survive every other reset and `setup.ps1` skips the HF download when they're present, saving you ~5-10 min per variant.

Light reset (rebuild venv + vendor only, models and caches preserved):

```powershell
Remove-Item -Recurse -Force .venv, vendor -ErrorAction SilentlyContinue
.\setup.ps1
```

This is the right reset for most cases: package install errors, weird import failures, stale editable installs. Total cost ~5 min (mostly the torch wheel re-download).

Full nuke (only if you actually need fresh weights too):

```powershell
Remove-Item -Recurse -Force .venv, vendor, models, outputs, .serve-logs -ErrorAction SilentlyContinue
.\setup.ps1
```

Adds ~5-10 min per model variant on top, since HF re-download is bandwidth-bound. If the download itself is slow, see the "HF downloads are slow" entry above.

## What setup.ps1 actually does on Windows

Reference, in case you want to do parts manually:

1. Checks `Is-Windows`.
2. **Preflight** (fails fast with a URL if anything's wrong):
   - x64 architecture
   - `git --version` on PATH
   - Free disk on the repo's drive (>= 15 GB hard fail, < 25 GB advisory)
   - NVIDIA driver >= 566.07 (advisory; older drivers will silently land torch in CPU mode)
   - Windows long-paths registry key (advisory; matters once Triton's kernel cache grows)
3. Installs uv (via Astral's PowerShell installer) if missing.
4. Creates `.venv` with Python 3.11 via `uv venv`.
5. Clones `vendor/image-studio` and `vendor/mflux-prism` from GitHub.
6. Patches `vendor/image-studio/pyproject.toml` to swap the mflux git pin for the local `vendor/mflux-prism` path.
7. Runs `uv sync --inexact` (gets `huggingface-hub`, `hf-transfer`, `nodejs-wheel-binaries`; everything else is gated to darwin/linux in pyproject.toml so it's skipped). The `--inexact` flag keeps `uv sync` from pruning the Windows GPU stack on subsequent re-runs, since those packages aren't in `uv.lock`.
8. **Windows GPU stack** (in `uv pip install` mode, not `uv sync`):
   - `torch==2.11.*` from `https://download.pytorch.org/whl/cu128`
   - `triton-windows>=3.6,<3.7` (provides `import triton` natively)
   - `gemlite hqq --no-deps` (skips the linux `triton` PyPI distribution)
   - `tqdm termcolor einops fastapi 'uvicorn[standard]' pydantic pillow 'diffusers>=0.38' transformers accelerate safetensors`
   - editable install of `vendor/image-studio/backend_gpu` with `--no-deps`
   - smoke test: `import torch, triton, gemlite, hqq, diffusers, transformers, accelerate; from backend_gpu.pipeline_gpu import GpuPipeline`
9. Wires Node: writes `node.cmd`, `npm.cmd`, `npx.cmd` shims into `.venv\Scripts\` that point at `node.exe` + `npm-cli.js` inside the `nodejs_wheel` site-packages dir.
10. If `$env:BONSAI_TOKEN` is set, runs `huggingface_hub.login(...)`.
11. Calls `.\scripts\download_model.ps1 $variant` unless `$env:SKIP_DOWNLOAD = '1'`.

## Knobs

| env var | default | what it does |
|---|---|---|
| `BONSAI_TOKEN` | unset | HF token; needed until launch |
| `BONSAI_VARIANT` | `ternary` | which Bonsai arm to download/serve (`ternary` or `binary`) |
| `SKIP_DOWNLOAD` | unset | `1` skips the model download in setup.ps1 |
| `BONSAI_SKIP_GPU_STACK` | unset | `1` skips the torch/triton/gemlite/hqq install (frontend-only setup, useful when the backend runs elsewhere) |
| `BONSAI_PACKAGE_MIN_AGE_DAYS` | `7` | min age in days for any uv- or npm-installed package version (supply-chain defense). `0` to disable |
| `BACKEND_PORT` | `8000` | port serve.ps1 binds the backend to |
| `FRONTEND_PORT` | `3000` | port serve.ps1 binds the frontend to |
| `BACKEND_READY_TIMEOUT` | `180` | seconds serve.ps1 waits for the backend's `/` to respond |
| `BONSAI_FRONTEND_PROD` | unset | `1` makes serve.ps1 run `next build && next start` instead of `next dev` |
| `NEXT_PUBLIC_BACKEND_URL` | `http://127.0.0.1:$BACKEND_PORT` | URL the Next.js frontend points at |

## Versions known to work (May 2026)

If something starts breaking after a fresh setup, the most likely cause is upstream drift. The exact versions this was last validated against:

- Windows 11 26200, PowerShell 5.1
- Python 3.11.9 (the version uv fetched)
- uv 0.11.1
- NVIDIA driver 566.07 (CUDA 12.8 runtime)
- torch 2.11.0+cu128
- triton-windows 3.6.0.post26
- gemlite 0.5.1.post1
- hqq 0.2.8.post1 (its CUDA C++ extension auto-skipped, fine)
- diffusers 0.38.0
- transformers 5.9.0
- accelerate 1.13.0
- node 26.1.0 + npm 11.13.0 (from `nodejs-wheel-binaries`)

## Reporting bugs

If something here breaks on your box, the most useful info to paste in an issue:

```powershell
# basic environment
$PSVersionTable.PSVersion
[Environment]::OSVersion.Version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Python stack
.\.venv\Scripts\python.exe -c "import sys, torch, triton, gemlite, hqq; print(sys.version); print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); print('triton', triton.__version__); print('gemlite', gemlite.__version__); print('hqq', hqq.__version__)"

# last lines of whichever log file was running when it broke
Get-Content .\.serve-logs\backend.log.err -Tail 50
Get-Content .\.serve-logs\frontend.log     -Tail 50
```
