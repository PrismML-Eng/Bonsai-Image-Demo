# Bonsai Image Demo -- Windows setup (PowerShell parallel to setup.sh).
#
# Installs the full native Windows inference stack on an NVIDIA GPU:
#   - uv + Python 3.11 venv
#   - torch 2.11 + CUDA 12.8
#   - triton-windows (drop-in for 'triton', built by the triton-lang org)
#   - gemlite + hqq (pyproject.toml gates these to linux; we install with
#     --no-deps on top of triton-windows)
#   - vendor/image-studio/backend_gpu (editable)
#   - nodejs-wheel-binaries for the Next.js studio frontend
#
# No WSL2 required. Tested on RTX 3060 Laptop + driver 566.07 + Win11.
#
# Usage:
#   .\setup.ps1
#
# IMPORTANT: Windows blocks script execution by default. ONE-TIME setup
# in any PowerShell window (or in your profile):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# Or run any individual script with a per-invocation bypass:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# Knobs (env vars):
#   BONSAI_VARIANT               'ternary' (default) or 'binary'.
#   SKIP_DOWNLOAD                '1' to skip the post-install model download.
#   BONSAI_SKIP_GPU_STACK        '1' to skip the torch/triton/gemlite/hqq
#                                install (frontend-only setup, e.g. when
#                                the backend will run on another box).
#   BONSAI_PACKAGE_MIN_AGE_DAYS  Min age in days for any installed package
#                                (default 7; '0' to disable).

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# ── Resolve paths ──
$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir
. (Join-Path $ScriptDir 'scripts\common.ps1')

$VenvDir       = Join-Path $ScriptDir '.venv'
$VenvPy        = Join-Path $VenvDir   'Scripts\python.exe'
$PythonVersion = '3.11'
$UvMin         = '0.7.0'

# Semver comparison: $true if $A >= $B. Tolerates pre-release suffixes
# (e.g. "0.11.1+local") by splitting on the first non-numeric char.
function Test-VersionGe {
    param([string]$A, [string]$B)
    function _parts([string]$v) {
        $v = ($v -split '[^\d.]', 2)[0]   # drop any '+local' / '-rc1' tail
        return $v.Split('.') | ForEach-Object { [int]$_ }
    }
    $pa = _parts $A
    $pb = _parts $B
    $n  = [Math]::Max($pa.Length, $pb.Length)
    for ($i = 0; $i -lt $n; $i++) {
        $ai = if ($i -lt $pa.Length) { $pa[$i] } else { 0 }
        $bi = if ($i -lt $pb.Length) { $pb[$i] } else { 0 }
        if ($ai -gt $bi) { return $true }
        if ($ai -lt $bi) { return $false }
    }
    return $true
}

Write-Host ""
Write-Host "========================================="
Write-Host "   Bonsai Image Demo Setup (Windows)"
Write-Host "========================================="
Write-Host ""

# ────────────────────────────────────────────────────
#  1. Platform sanity
# ────────────────────────────────────────────────────
if (-not (Is-Windows)) {
    err "setup.ps1 is the Windows setup script. On macOS/Linux use ./setup.sh."
    exit 1
}

# Cross-version arch detection. RuntimeInformation.OSArchitecture requires
# .NET Framework 4.7.1+, which Windows PowerShell 5.1 boxes may not have --
# the property returns $null and our preflight then fails with a confusing
# empty "Unsupported architecture: " message. Fall back to $env vars which
# are populated by the kernel on every Windows host.
$arch = $null
try {
    $a = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    if ($a) { $arch = $a.ToString() }
} catch { }
if (-not $arch) {
    if ([Environment]::Is64BitOperatingSystem) {
        # PROCESSOR_ARCHITEW6432 is set when a 32-bit process runs on a 64-bit
        # OS (WOW64). Otherwise PROCESSOR_ARCHITECTURE is the OS native arch.
        $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    } else {
        $arch = $env:PROCESSOR_ARCHITECTURE
    }
    if (-not $arch) { $arch = 'unknown' }
}
step "Detected platform: Windows ($arch)"

# ────────────────────────────────────────────────────
#  1b. Preflight: things we don't install but that setup needs
# ────────────────────────────────────────────────────
# Each check produces either a hard fail (with a URL + FAQ pointer) or a
# soft warn. Order matters: cheapest + most-likely-to-fail first.
step "Preflight checks ..."

$preflightFatal = $false

# --- Architecture: triton-windows + torch cu128 wheels are x64-only.
# Accept the common 64-bit spellings (-in is case-insensitive in PS).
if ($arch -notin 'X64','Amd64','AMD64') {
    err "Unsupported architecture: $arch. triton-windows and torch cu128 wheels are x64 only."
    $preflightFatal = $true
}

# --- Git on PATH: needed for the vendor/ clones in step 5.
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    err "git not found on PATH."
    Write-Host "       setup.ps1 needs git to clone vendor/image-studio and vendor/mflux-prism."
    Write-Host "       Install Git for Windows from https://git-scm.com/download/win, then re-run."
    Write-Host "       See scripts/windows.md FAQ ('git is not recognized') for details."
    $preflightFatal = $true
} else {
    $gitVer = (& $gitCmd.Source --version 2>$null) -replace '^git\s+version\s+', ''
    info "git $gitVer at $($gitCmd.Source)"
}

# --- Free disk space: torch wheel 2.6GB + model 4GB + node_modules + caches.
# Resolve which drive the repo lives on (not always C:).
$repoRoot = Split-Path -Qualifier $ScriptDir
$drive    = Get-PSDrive -Name ($repoRoot.TrimEnd(':')) -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    if ($env:BONSAI_SKIP_GPU_STACK -eq '1') {
        # Without GPU stack: no torch (2.6 GB). Only venv + Node.js + model (~6 GB).
        if ($freeGB -lt 6) {
            err "Only ${freeGB} GB free on $repoRoot drive; setup needs ~6 GB (model 4 GB, frontend deps, caches)."
            $preflightFatal = $true
        } elseif ($freeGB -lt 10) {
            warn "${freeGB} GB free on $repoRoot drive. Tight on caches without GPU stack; 10 GB+ recommended."
        } else {
            info "Free disk on $repoRoot drive: ${freeGB} GB"
        }
    } else {
        if ($freeGB -lt 15) {
            err "Only ${freeGB} GB free on $repoRoot drive; setup needs ~15 GB (torch 2.6 GB, model 4 GB, frontend deps, caches)."
            $preflightFatal = $true
        } elseif ($freeGB -lt 25) {
            warn "${freeGB} GB free on $repoRoot drive. Setup will succeed but you'll be tight on caches; 25 GB+ recommended."
        } else {
            info "Free disk on $repoRoot drive: ${freeGB} GB"
        }
    }
}

# --- Long path support: Triton's JIT cache writes under outputs/.triton_cache/
# and easily exceeds the 260-char MAX_PATH on deeply-nested clones.
# Advisory: not fatal if disabled because we may not hit it, but the FAQ
# explains the symptom.
try {
    $lp = Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction Stop
    if ($lp.LongPathsEnabled -ne 1) {
        warn "Windows long-path support is OFF. Triton's kernel cache may hit MAX_PATH for deep clones."
        Write-Host "       Fix (elevated, one-time + reboot):"
        Write-Host "         New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force"
        Write-Host "       Or move the repo closer to the drive root (C:\Bonsai instead of C:\Users\you\Desktop\...)."
    } else {
        info "Long-path support: enabled"
    }
} catch {
    # Key doesn't exist on older Win10 builds; same fix as the off branch.
    warn "Couldn't read LongPathsEnabled registry. If you hit path-length errors during inference, see scripts/windows.md."
}

# --- NVIDIA driver + GPU detection. We don't gate setup on GPU presence
# (frontend-only install is still useful), but we DO check driver version
# when a GPU is found; cu128 wheels are silent CPU-only on older drivers.
$gpuName     = ''
$driverVer   = ''
$driverOk    = $false
if (Has-NvidiaGpu) {
    try {
        $smiOut = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
        if ($smiOut) {
            $first = ($smiOut | Select-Object -First 1).ToString().Trim()
            $parts = $first -split ',\s*', 2
            if ($parts.Count -ge 1) { $gpuName   = $parts[0].Trim() }
            if ($parts.Count -ge 2) { $driverVer = $parts[1].Trim() }
        }
    } catch { }
    if (-not $gpuName) {
        # WMI fallback for GPU name when nvidia-smi flakes out (rare but seen
        # on fresh installs where the driver pkg is half-installed).
        try {
            $card = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1
            if ($card) { $gpuName = $card.Name }
        } catch { }
    }
    if (-not $gpuName) { $gpuName = 'NVIDIA GPU' }

    # Driver version compare: 566.07+ is what torch cu128 wheels expect.
    if ($driverVer -match '^(\d+)\.(\d+)') {
        $maj = [int]$Matches[1]
        $min = [int]$Matches[2]
        if ($maj -gt 566 -or ($maj -eq 566 -and $min -ge 7)) {
            $driverOk = $true
        }
    }

    if ($driverVer) {
        if ($driverOk) {
            info "NVIDIA GPU: $gpuName, driver $driverVer (CUDA 12.8 capable)"
        } else {
            warn "NVIDIA GPU: $gpuName, driver $driverVer -- below 566.07. torch cu128 may fall back to CPU."
            Write-Host "       Update from https://www.nvidia.com/Download/index.aspx and reboot, then re-run."
            Write-Host "       See scripts/windows.md ('torch+cu128 install failed') for details."
        }
    } else {
        warn "NVIDIA GPU detected ($gpuName) but driver version unreadable."
    }
} else {
    warn "No NVIDIA GPU detected. Setup will install the frontend only (no inference)."
    Write-Host "       Set `$env:BONSAI_SKIP_GPU_STACK = '1' to silence this and skip the GPU install block."
}

if ($preflightFatal) {
    err "Preflight checks failed. Fix the items above and re-run setup.ps1."
    exit 1
}
Write-Host ""

# pyproject.toml gates gemlite/hqq/backend_gpu to sys_platform=='linux', so
# uv sync alone produces a frontend-only env on Windows. The block at step
# 6.5 below adds the Windows GPU stack on top: torch+cu128, triton-windows
# (drop-in for 'triton'), then gemlite/hqq/backend_gpu via uv pip install
# with --no-deps to dodge the linux 'triton' PyPI distribution.

# ────────────────────────────────────────────────────
#  2. Check Python (uv will pick up an existing 3.11; otherwise it
#     downloads its own -- no system Python required)
# ────────────────────────────────────────────────────
step "Checking system Python ..."
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pyCmd) {
    $pyVer = (& $pyCmd.Source --version 2>&1).ToString().Trim()
    info "$pyVer at $($pyCmd.Source)"
} else {
    warn "No system Python on PATH -- uv will fetch its own $PythonVersion."
}

# ────────────────────────────────────────────────────
#  3. Install uv
# ────────────────────────────────────────────────────
function Test-Uv {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) { return $false }
    $ver = (& $uv.Source --version 2>$null) -replace '^uv\s+', '' -replace '\s.*$', ''
    if (-not $ver) { return $false }
    return Test-VersionGe $ver $UvMin
}

step "Checking uv ..."
if (Test-Uv) {
    # Precompute to dodge PS 5.1's parser choking on chained -replace inside $().
    $uvVer = (& uv --version) -replace '^uv\s+', '' -replace '\s.*$', ''
    info "uv $uvVer found."
} else {
    step "Installing uv ..."
    # Astral's official Windows installer. Writes uv.exe to
    # %USERPROFILE%\.local\bin and modifies the user PATH; we re-prepend
    # it for the current session so subsequent commands find it.
    $oldProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-RestMethod 'https://astral.sh/uv/install.ps1' | Invoke-Expression
    } finally {
        $ProgressPreference = $oldProgress
    }
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Test-Uv)) {
        err "uv install failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    }
    info "uv installed."
}

# ────────────────────────────────────────────────────
#  4. Create Python venv
# ────────────────────────────────────────────────────
step "Setting up Python environment ..."
if (Test-Path $VenvPy) {
    info "Existing venv found at $VenvDir"
} else {
    & uv venv $VenvDir --python $PythonVersion
    if ($LASTEXITCODE -ne 0) { err "uv venv failed."; exit 1 }
    info "Created venv with Python $PythonVersion"
}

# ────────────────────────────────────────────────────
#  5. Clone private deps into vendor/
# ────────────────────────────────────────────────────
$VendorDir = Join-Path $ScriptDir 'vendor'
New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null

function Invoke-CloneVendor {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Url,
        [string]$Branch
    )
    $dest = Join-Path $VendorDir $Name
    if (Test-Path (Join-Path $dest '.git')) {
        info "vendor/$Name already cloned -- skipping (git pull manually to update)."
        return
    }
    $branchSuffix = if ($Branch) { " ($Branch)" } else { '' }
    step "Cloning $Name$branchSuffix into vendor/ ..."
    if ($Branch) {
        & git clone --branch $Branch $Url $dest
    } else {
        & git clone $Url $dest
    }
    if ($LASTEXITCODE -ne 0) { err "git clone $Name failed."; exit 1 }
}

Invoke-CloneVendor -Name 'image-studio' -Url 'https://github.com/PrismML-Eng/image-studio.git'
Invoke-CloneVendor -Name 'mflux-prism'  -Url 'https://github.com/PrismML-Eng/mflux-prism.git'

# image-studio's pyproject still pins mflux to a git rev -- same patch as
# setup.sh, idempotent: only touches the file if the git source is still there.
$studioPp = Join-Path $VendorDir 'image-studio\pyproject.toml'
if ((Test-Path $studioPp) -and ((Get-Content $studioPp -Raw) -match '(?m)^mflux = \{ git = ')) {
    step "Patching vendor/image-studio mflux source to match vendor layout ..."
    $content = Get-Content $studioPp -Raw
    $patched = $content -replace `
        '(?m)^mflux = \{ git = .*$', `
        'mflux = { path = "../mflux-prism", editable = true }'
    Set-Content -Path $studioPp -Value $patched -Encoding UTF8 -NoNewline
}

# ────────────────────────────────────────────────────
#  6. uv sync
# ────────────────────────────────────────────────────
# On Windows this resolves to:
#   nodejs-wheel-binaries, huggingface-hub, hf-transfer
# Everything else is gated by `sys_platform == 'darwin'` or `== 'linux'`
# in pyproject.toml and is silently skipped. No retry-on-mlx-cache dance
# needed; that's a macOS-only failure mode.
step "Running uv sync ..."
$pkgCutoff = Get-PackageCutoffDate
# --inexact: don't prune packages that aren't in uv.lock. The Windows GPU
# stack (torch, triton-windows, gemlite, hqq, ...) is installed in the
# next block via `uv pip install`, not tracked in the lock, so a plain
# `uv sync` on the second run would helpfully uninstall the entire GPU
# stack just to bring the venv "back into sync" with the lock. --inexact
# stops that. The lock-tracked deps are still resolved + updated as normal.
& uv sync --inexact --exclude-newer $pkgCutoff
if ($LASTEXITCODE -ne 0) {
    err "uv sync failed (cutoff $pkgCutoff). Re-run with BONSAI_PACKAGE_MIN_AGE_DAYS=0 if a recent dep is needed."
    exit 1
}
info "uv sync complete (versions <= $pkgCutoff)."

# ────────────────────────────────────────────────────
#  6.5 Windows GPU stack (torch+CUDA, triton-windows, gemlite, hqq)
# ────────────────────────────────────────────────────
# pyproject.toml's `sys_platform == 'linux'` markers keep `uv sync` from
# touching gemlite/hqq/backend_gpu on Windows. We install them as a
# follow-up uv pip pass.
#
# Key tricks:
#   - triton-windows installs the top-level `triton` module, so gemlite's
#     `import triton` resolves natively.
#   - gemlite + hqq go in with --no-deps so their `triton>=3.6` requirement
#     does NOT pull in the linux 'triton' PyPI distribution (which fails
#     to build on win32).
#   - backend_gpu is editable-installed from vendor/image-studio/backend_gpu
#     with --no-deps; we install its real runtime deps explicitly.
#
# Skippable with $env:BONSAI_SKIP_GPU_STACK='1' if you only want the
# frontend on this box (e.g. backend lives on another machine).
if (Has-NvidiaGpu -and ($env:BONSAI_SKIP_GPU_STACK -ne '1')) {
    step "Installing Windows GPU stack (torch+cu128, triton-windows, gemlite, hqq, backend_gpu) ..."

    # The torch wheel index pins a single torch matching cu128. The
    # constraint floor here matches the agent-tested compat matrix:
    # torch 2.11 pairs with triton 3.6 on cu128.
    & uv pip install --python $VenvPy `
        --index-url 'https://download.pytorch.org/whl/cu128' `
        'torch==2.11.*'
    if ($LASTEXITCODE -ne 0) { err "torch+cu128 install failed."; exit 1 }

    # triton-windows constrained to <3.7 so we don't accidentally pull a
    # version that wants torch 2.12 (which isn't on cu128 yet).
    & uv pip install --python $VenvPy 'triton-windows>=3.6,<3.7' numpy
    if ($LASTEXITCODE -ne 0) { err "triton-windows install failed."; exit 1 }

    # gemlite + hqq -- gemlite hard-requires `triton>=3.6` which would
    # resolve to the linux-only PyPI 'triton' on win32. --no-deps skips
    # the resolver; the install is fine because the top-level `triton`
    # module is already provided by triton-windows.
    & uv pip install --python $VenvPy --no-deps gemlite hqq
    if ($LASTEXITCODE -ne 0) { err "gemlite/hqq install failed."; exit 1 }

    # Their actual runtime deps (gemlite needs numpy+tqdm; hqq pulls in
    # termcolor+einops). transformers/tokenizers are needed by
    # backend_gpu._load_text_encoder and _load_tokenizer.
    & uv pip install --python $VenvPy `
        tqdm termcolor einops `
        'fastapi>=0.115' 'uvicorn[standard]>=0.30' 'pydantic>=2.7' `
        'pillow>=10.4' 'diffusers>=0.38' transformers accelerate safetensors
    if ($LASTEXITCODE -ne 0) { err "backend_gpu runtime deps install failed."; exit 1 }

    # Editable install of the vendored backend_gpu package. --no-deps to
    # skip its declared `prism-image-studio-backend-gpu` deps which are
    # already covered above; uv otherwise re-resolves them.
    & uv pip install --python $VenvPy --no-deps -e (Join-Path $VendorDir 'image-studio\backend_gpu')
    if ($LASTEXITCODE -ne 0) { err "backend_gpu editable install failed."; exit 1 }

    # Confirm the whole stack imports together. Catches drift (e.g. uv
    # uninstalling something while resolving the next install).
    $smokeOk = & $VenvPy -c "import torch, triton, gemlite, hqq, diffusers, transformers, accelerate; from backend_gpu.pipeline_gpu import GpuPipeline; print('ok')" 2>&1
    if ($smokeOk -match '^ok') {
        $cudaState = & $VenvPy -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1
        info "Windows GPU stack ready ($cudaState)"
    } else {
        warn "Windows GPU stack installed but import smoke test failed:"
        Write-Host $smokeOk
    }
} elseif ($env:BONSAI_SKIP_GPU_STACK -eq '1') {
    info "Skipping Windows GPU stack (BONSAI_SKIP_GPU_STACK=1) -- frontend-only install."
} else {
    warn "No NVIDIA GPU detected -- skipping Windows GPU stack. Frontend will still run."
}

# ────────────────────────────────────────────────────
#  7. Wire bundled Node.js
# ────────────────────────────────────────────────────
# nodejs-wheel-binaries on Windows installs node.exe at
#   .venv\Lib\site-packages\nodejs_wheel\node.exe
# but doesn't drop a working npm.cmd next to it -- the bundled
# lib\node_modules\npm\bin\npm.cmd has the wrong relative path to npm-cli.js
# and will fail with "Cannot find module" if invoked directly. Mac/Linux
# setup.sh papers over this with symlinks; on Windows we write small cmd
# wrappers into .venv\Scripts\ that invoke node.exe + npm-cli.js (and
# npx-cli.js) with absolute paths.
step "Wiring bundled Node.js into .venv\\Scripts\\ ..."

$wheelDir = & $VenvPy -c "import os, nodejs_wheel; print(os.path.dirname(nodejs_wheel.__file__))" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $wheelDir) {
    err "nodejs_wheel not importable -- uv sync may have failed."
    exit 1
}

$nodeExe = Join-Path $wheelDir 'node.exe'
if (-not (Test-Path $nodeExe)) {
    err "node.exe not at $nodeExe -- check nodejs-wheel-binaries install."
    exit 1
}

# Path to the cli scripts inside the wheel's node_modules tree.
$npmCli = Join-Path $wheelDir 'lib\node_modules\npm\bin\npm-cli.js'
$npxCli = Join-Path $wheelDir 'lib\node_modules\npm\bin\npx-cli.js'

# Helper to write a small cmd wrapper at $dest that runs node + the given
# cli script. Using %* preserves quoted args. setlocal scopes any env
# tweaks the cli might do.
function Write-NpmShim {
    param([string]$Dest, [string]$Cli)
    if (-not (Test-Path $Cli)) {
        warn "skipping shim $Dest -- backing js not at $Cli"
        return
    }
    $script = @"
@echo off
setlocal
"$nodeExe" "$Cli" %*
"@
    Set-Content -Path $Dest -Value $script -Encoding ASCII -NoNewline
}

# Also write a node.exe shim in .venv\Scripts\ so PATH-prepending Scripts/
# gives callers a working `node`. Easiest: a one-line cmd that execs the
# real node.exe (a symlink would require admin or Developer Mode on
# Windows; cmd wrappers don't).
$nodeShim = Join-Path $VenvDir 'Scripts\node.cmd'
$nodeWrapper = @"
@echo off
"$nodeExe" %*
"@
Set-Content -Path $nodeShim -Value $nodeWrapper -Encoding ASCII -NoNewline

$npmShim = Join-Path $VenvDir 'Scripts\npm.cmd'
$npxShim = Join-Path $VenvDir 'Scripts\npx.cmd'
Write-NpmShim -Dest $npmShim -Cli $npmCli
Write-NpmShim -Dest $npxShim -Cli $npxCli

# Sanity: invoke each to confirm the wrappers work end-to-end.
$nodeVer = (& $nodeExe --version) 2>$null
$npmVer  = if (Test-Path $npmShim) { (& $npmShim --version) 2>$null } else { $null }
if ($nodeVer -and $npmVer) {
    info "node $nodeVer + npm $npmVer ready in .venv\Scripts\"
} elseif ($nodeVer) {
    warn "node $nodeVer ready, but npm shim failed to report a version."
} else {
    warn "Bundled Node.js wired, but `node --version` returned nothing."
}

# ────────────────────────────────────────────────────
#  8. Download the default model (skippable)
# ────────────────────────────────────────────────────
if ($env:SKIP_DOWNLOAD -ne '1') {
    $variant = if ($env:BONSAI_VARIANT) { $env:BONSAI_VARIANT } else { 'ternary' }
    step "Downloading default model: $variant (`$env:BONSAI_VARIANT=binary to switch, `$env:SKIP_DOWNLOAD=1 to skip)..."
    & (Join-Path $ScriptDir 'scripts\download_model.ps1') $variant
    if ($LASTEXITCODE -eq 0) {
        info "$variant model present."
    } else {
        warn "Model download failed. Retry with:"
        Write-Host "    .\scripts\download_model.ps1 $variant"
    }
}

# ────────────────────────────────────────────────────
#  Done!
# ────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================="
Write-Host "   Setup complete!"
Write-Host "========================================="
Write-Host ""
Write-Host "  Run the full studio (backend + frontend):"
Write-Host "    .\scripts\serve.ps1"
Write-Host ""
Write-Host "  Once it's up, generate from another terminal:"
Write-Host "    .\scripts\send_request.ps1 -p `"a tiny bonsai tree in a ceramic pot`""
Write-Host ""
Write-Host "  Or, one-shot CLI without a server (pays cold-start every call):"
Write-Host "    .\scripts\generate.ps1 -p `"a tiny bonsai tree in a ceramic pot`""
Write-Host ""
