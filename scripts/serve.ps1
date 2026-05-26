# Launch the Bonsai Image studio: image-studio's FastAPI backend on :8000
# (with this repo's models/ tree) plus the Next.js frontend on :3000. Both
# run in the foreground; Ctrl+C tears them down together.
#
# Native Windows runs the same backend that Linux does: gemlite + hqq +
# triton kernels via triton-windows. setup.ps1 installs the stack; this
# script just boots uvicorn + npm dev.
#
# Env knobs (match serve.sh):
#   BACKEND_PORT             Override 8000.
#   FRONTEND_PORT            Override 3000.
#   STUDIO_DIR               Path to image-studio checkout
#                            (default: vendor/image-studio).
#   BACKEND_READY_TIMEOUT    Seconds to wait for /backends to answer
#                            (default 180). Bump on slow GPUs.
#   BONSAI_VARIANT           'ternary' (default) or 'binary'.
#   BONSAI_FRONTEND_PROD=1   `next build` + `next start` instead of `next dev`.
#   NEXT_PUBLIC_BACKEND_URL  Override the URL the frontend hits (default
#                            http://127.0.0.1:$BACKEND_PORT).

[CmdletBinding()]
param(
    [switch]$FrontendOnly,
    [switch]$BackendOnly
)

$ErrorActionPreference = 'Stop'

$DemoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Venv $DemoDir

$BackendPort  = if ($env:BACKEND_PORT)  { $env:BACKEND_PORT }  else { '8000' }
$FrontendPort = if ($env:FRONTEND_PORT) { $env:FRONTEND_PORT } else { '3000' }
$StudioDir    = if ($env:STUDIO_DIR)    { $env:STUDIO_DIR }    else { (Join-Path $DemoDir 'vendor\image-studio') }
$BackendUrl   = if ($env:NEXT_PUBLIC_BACKEND_URL) { $env:NEXT_PUBLIC_BACKEND_URL } else { "http://127.0.0.1:$BackendPort" }
$BackendReadyTimeout = if ($env:BACKEND_READY_TIMEOUT) { [int]$env:BACKEND_READY_TIMEOUT } else { 180 }

# ── Platform gate ──
if (-not (Is-Windows)) {
    err "serve.ps1 is the Windows entry. On macOS/Linux use ./scripts/serve.sh."
    exit 1
}

if (Has-NvidiaGpu) {
    info "Platform: Windows + NVIDIA GPU -- backend_gpu (gemlite/HQQ on CUDA, via triton-windows)"
} else {
    warn "Platform: Windows without NVIDIA GPU -- backend_gpu needs CUDA, generation will fail."
    Write-Host "       Install an NVIDIA driver (566.07+ recommended for CUDA 12.8 wheels)."
    Write-Host "       Continuing anyway."
}

# ── Variant + model path resolution ──
$Variant = if ($env:BONSAI_VARIANT) { $env:BONSAI_VARIANT } else { 'ternary' }
if ($Variant -notin 'ternary','binary') {
    err "BONSAI_VARIANT must be 'ternary' or 'binary' (got $Variant)"
    exit 1
}
$DefaultBackend = "bonsai-$Variant-gemlite"

# backend_gpu/pipeline_gpu.py reads SEPARATE env vars per variant:
#   MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH  for bonsai-ternary-gemlite
#   MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH   for bonsai-binary-gemlite
# If a variant's path is unset and a /generate request targets that arm
# (e.g. a client that omits `backend` -- the GenerateRequest default is
# bonsai-binary-gemlite), the pipeline falls back to the hardcoded
# /root/models/bonsai-... default and FileNotFoundErrors. So we scan
# models/ for BOTH variants and set whichever env vars correspond to
# what is actually on disk. Result: /generate works for any backend the
# user has downloaded, no matter which BONSAI_VARIANT was loaded at boot.
function Resolve-TransformerDir {
    param([string]$VariantName)
    $dir = Join-Path $DemoDir "models\bonsai-image-4B-$VariantName-gemlite"
    if (-not (Test-Path $dir)) { return $null }
    $hit = Get-ChildItem -Path $dir -Directory -Filter 'transformer-gemlite-*' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { return $hit.FullName } else { return $null }
}

$TernaryTransformerDir = Resolve-TransformerDir 'ternary'
$BinaryTransformerDir  = Resolve-TransformerDir 'binary'

# The active variant (the one we'll load at boot) MUST be present on disk.
# The other variant is optional; if present we still wire the env var so
# /generate calls targeting it work, otherwise we leave it unset (and a
# request that targets it gets a clean FileNotFoundError instead of the
# silent /root/models/... default).
$ModelDir = Join-Path $DemoDir "models\bonsai-image-4B-$Variant-gemlite"
$ActiveTransformerDir = if ($Variant -eq 'binary') { $BinaryTransformerDir } else { $TernaryTransformerDir }
if (-not $ActiveTransformerDir) {
    err "no transformer-gemlite-* subdir found under $ModelDir"
    Write-Host "       Download the model first: .\scripts\download_model.ps1 $Variant"
    exit 1
}

# Report what we found, since this is the common confusing failure point.
$bothMsg = @()
if ($TernaryTransformerDir) { $bothMsg += "ternary at $TernaryTransformerDir" } else { $bothMsg += "ternary NOT downloaded" }
if ($BinaryTransformerDir)  { $bothMsg += "binary at $BinaryTransformerDir" }   else { $bothMsg += "binary NOT downloaded" }
info ("Transformer pool: " + ($bothMsg -join "; "))

# ── Resolve frontend dir ──
if (-not (Test-Path $StudioDir)) {
    err "image-studio not found at $StudioDir"
    Write-Host "       Run .\setup.ps1 to clone it into vendor\, or set `$env:STUDIO_DIR."
    exit 1
}
$FrontendDir = Join-Path $StudioDir 'frontend'
if (-not (Test-Path $FrontendDir)) {
    err "frontend not found at $FrontendDir"
    exit 1
}

# ── Port-in-use check ──
# Get-NetTCPConnection is the native Windows equivalent of lsof for sockets.
function Test-PortInUse {
    param([int]$Port)
    try {
        $hit = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        return [bool]$hit
    } catch {
        return $false
    }
}
if (-not $FrontendOnly -and (Test-PortInUse $BackendPort)) {
    err "Port $BackendPort already in use (backend)."
    exit 1
}
if (-not $BackendOnly -and (Test-PortInUse $FrontendPort)) {
    err "Port $FrontendPort already in use (frontend)."
    exit 1
}

# ── npm shim setup (only needed if running the frontend) ──
$VenvBin = Get-VenvBin $DemoDir
$NpmCmd  = Join-Path $VenvBin 'npm.cmd'
if (-not $BackendOnly) {
    if (-not (Test-Path $NpmCmd)) {
        err "Bundled npm not found at $NpmCmd - did .\setup.ps1 run successfully?"
        exit 1
    }
    $env:PATH = "$VenvBin;$env:PATH"
}

# ── Install frontend deps on first run ──
if (-not $BackendOnly) {
    $NodeModules = Join-Path $FrontendDir 'node_modules'
    if (-not (Test-Path $NodeModules)) {
        $pkgCutoff = Get-PackageCutoffDate
        step "Installing frontend dependencies (first run, versions <= $pkgCutoff)..."
        Push-Location $FrontendDir
        try {
            & $NpmCmd install --no-audit --no-fund --before $pkgCutoff
            if ($LASTEXITCODE -ne 0) {
                err "npm install failed."
                exit 1
            }
        } finally {
            if ((Get-Location).Path -eq (Resolve-Path $FrontendDir).Path) {
                Pop-Location
            }
        }
    }
}

# ── Logs ──
$LogDir = Join-Path $DemoDir '.serve-logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$BackendLog  = Join-Path $LogDir 'backend.log'
$FrontendLog = Join-Path $LogDir 'frontend.log'

# ── Production build (optional, frontend) ──
$useProd = $env:BONSAI_FRONTEND_PROD -eq '1'
$DotNext = Join-Path $FrontendDir '.next'
if (-not $BackendOnly -and $useProd -and -not (Test-Path $DotNext)) {
    step "Building frontend (production, BONSAI_FRONTEND_PROD=1) -- first run only ..."
    Push-Location $FrontendDir
    try {
        $env:NEXT_PUBLIC_BACKEND_URL = $BackendUrl
        & $NpmCmd run build
        if ($LASTEXITCODE -ne 0) {
            err "frontend build failed."
            exit 1
        }
    } finally {
        if ((Get-Location).Path -eq (Resolve-Path $FrontendDir).Path) {
            Pop-Location
        }
    }
}

# ── Helpers ──
function Wait-ForPort {
    param(
        [int]$Port,
        [string]$Name,
        [int]$ProcessId,
        [int]$MaxSeconds = 60,
        [string]$LogPath = $null
    )
    Write-Host "       waiting for $Name on :$Port (timeout ${MaxSeconds}s) ..."
    # Heartbeat: emit a status line every $heartbeatEvery seconds so the user
    # knows we're still alive. If a log file was passed, also surface the
    # latest non-empty line from it -- that's where the model-loader prints
    # its stages ("loading gemlite transformer", "loaded text encoder in
    # 5.52s", etc.), so the user can see real progress, not just "waiting".
    $heartbeatEvery = 5
    $lastHeartbeat  = 0
    $lastLogLine    = ''
    for ($i = 0; $i -lt $MaxSeconds; $i++) {
        # Bare TCP connect rather than an HTTP probe. backend_gpu has no
        # GET / route (it returns 404), and Invoke-WebRequest throws on
        # 4xx in PS 5.1, so an HTTP probe would never see "ready" even
        # after uvicorn is up. A successful TCP handshake means uvicorn
        # is listening, which is exactly the signal we want.
        $tcp = New-Object System.Net.Sockets.TcpClient
        try {
            $iar = $tcp.BeginConnect('127.0.0.1', $Port, $null, $null)
            if ($iar.AsyncWaitHandle.WaitOne(1000) -and $tcp.Connected) {
                $tcp.EndConnect($iar) | Out-Null
                info "$Name ready on :$Port (took ${i}s)"
                return 0
            }
        } catch {
            # Connection refused / reset -- uvicorn not bound yet. Fall through.
        } finally {
            $tcp.Close()
        }
        # Bail early if the process is gone.
        $p = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $p) { return 1 }

        if ($i -gt 0 -and ($i - $lastHeartbeat) -ge $heartbeatEvery) {
            $lastHeartbeat = $i
            $stage = ''
            if ($LogPath -and (Test-Path $LogPath)) {
                # Latest informative line: prefer the last log line that looks
                # like a logger event ("YYYY-MM-DD HH:MM:SS LEVEL ..." or
                # uvicorn's "INFO:") and trim it to keep the heartbeat one row.
                try {
                    $tail = Get-Content -LiteralPath $LogPath -Tail 20 -ErrorAction Stop
                    $informative = $tail | Where-Object {
                        $_ -match '^\d{4}-\d{2}-\d{2}\s' -or $_ -match '^(INFO|WARNING|ERROR):'
                    } | Select-Object -Last 1
                    if ($informative) { $stage = $informative.ToString().Trim() }
                } catch { }
            }
            if ($stage) {
                # Squash the timestamp + logger name prefix so the heartbeat
                # is readable in one terminal row.
                $stageShort = $stage -replace '^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},?\d*\s', '' `
                                     -replace '^(INFO|WARNING|ERROR)\s+[^:]+:\s+', ''
                if ($stageShort.Length -gt 140) { $stageShort = $stageShort.Substring(0,137) + '...' }
                # Only re-print if the stage has changed; otherwise just bump the timer.
                if ($stageShort -ne $lastLogLine) {
                    Write-Host ("       [{0,3}s] {1}" -f $i, $stageShort)
                    $lastLogLine = $stageShort
                } else {
                    Write-Host ("       [{0,3}s] still loading ..." -f $i)
                }
            } else {
                Write-Host ("       [{0,3}s] still waiting on $Name ..." -f $i)
            }
        }

        Start-Sleep -Seconds 1
    }
    return 2
}

function Stop-ProcessTree {
    param([int]$ProcessId)
    try { & taskkill.exe /F /T /PID $ProcessId 2>&1 | Out-Null } catch { }
}

function Stop-PortListeners {
    param([int]$Port)
    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        foreach ($c in $conns) { Stop-ProcessTree -ProcessId $c.OwningProcess }
    } catch { }
}

# ── Start backend ──
$backendProcess = $null
if (-not $FrontendOnly) {
    step "Starting backend on :$BackendPort (default arm: $DefaultBackend)"
    Write-Host "       logs: $BackendLog"

    # uvicorn driving scripts.local_backend:app from $DemoDir so the
    # dotted module resolves (no sys.path tricks needed). The env vars
    # below match what backend_gpu/pipeline_gpu.py reads at lifespan time
    # to locate the four artifact dirs.
    $venvPy     = Get-VenvPython $DemoDir
    $uvicornExe = Join-Path $VenvBin 'uvicorn.exe'
    if (-not (Test-Path $uvicornExe)) {
        # Fall back to `python -m uvicorn` if the shim isn't there.
        $uvicornExe = $venvPy
    }

    # Per-process env block so we don't permanently mutate the parent
    # shell's MFLUX_STUDIO_GPU_* vars. Start-Process inherits the current
    # session env; set on $env: just before launch, restore after.
    $envKeys = @(
        'MFLUX_STUDIO_GPU_DEFAULT_BACKEND',
        'MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH',
        'MFLUX_STUDIO_GPU_VAE_PATH',
        'MFLUX_STUDIO_GPU_TOKENIZER_PATH',
        'MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH',
        'MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH',
        'PYTHONIOENCODING',
        'PYTHONUTF8'
    )
    $prevEnv = @{}
    foreach ($k in $envKeys) {
        $prevEnv[$k] = [Environment]::GetEnvironmentVariable($k, 'Process')
    }

    $env:MFLUX_STUDIO_GPU_DEFAULT_BACKEND     = $DefaultBackend
    # text_encoder / vae / tokenizer are loaded ONCE at boot and are
    # identical across variants (same Qwen-3 + Flux-2 VAE), so point them
    # at whichever variant the user has loaded.
    $env:MFLUX_STUDIO_GPU_TEXT_ENCODER_PATH   = Join-Path $ModelDir 'text_encoder-hqq-4bit'
    $env:MFLUX_STUDIO_GPU_VAE_PATH            = Join-Path $ModelDir 'vae'
    $env:MFLUX_STUDIO_GPU_TOKENIZER_PATH      = Join-Path (Join-Path $ModelDir 'text_encoder-hqq-4bit') 'tokenizer'

    # Set BOTH transformer env vars to local Windows paths even when the
    # corresponding model isn't downloaded yet. Two cases:
    #   - dir exists on disk -> point at the resolved transformer-gemlite-*
    #     so /generate for that arm works.
    #   - dir missing       -> point at where it WOULD live under models/
    #     so /generate for that arm fails with a FileNotFoundError that
    #     names the actual local path (an actionable hint), instead of
    #     pipeline_gpu.py's hardcoded /root/models/bonsai-binary fallback.
    $ternaryPath = if ($TernaryTransformerDir) {
        $TernaryTransformerDir
    } else {
        Join-Path $DemoDir 'models\bonsai-image-4B-ternary-gemlite\transformer-gemlite-int2'
    }
    $binaryPath = if ($BinaryTransformerDir) {
        $BinaryTransformerDir
    } else {
        Join-Path $DemoDir 'models\bonsai-image-4B-binary-gemlite\transformer-gemlite-int1'
    }
    [Environment]::SetEnvironmentVariable('MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH', $ternaryPath, 'Process')
    [Environment]::SetEnvironmentVariable('MFLUX_STUDIO_GPU_BINARY_TRANSFORMER_PATH',  $binaryPath,  'Process')

    # backend_gpu logs go through Python stdlib `logging` -> stderr by default,
    # but any incidental print() that hits non-ASCII would crash on Windows cp1252.
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8       = '1'

    if ($uvicornExe -eq $venvPy) {
        $backendArgs = @('-m','uvicorn','scripts.local_backend:app','--port',$BackendPort)
    } else {
        $backendArgs = @('scripts.local_backend:app','--port',$BackendPort)
    }

    $backendProcess = Start-Process `
        -FilePath $uvicornExe `
        -ArgumentList $backendArgs `
        -WorkingDirectory $DemoDir `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $BackendLog `
        -RedirectStandardError  "$BackendLog.err"

    # Restore the previous values so the parent shell isn't permanently
    # tainted with paths from a backend launch.
    foreach ($k in $prevEnv.Keys) {
        if ($null -eq $prevEnv[$k]) {
            Remove-Item ("Env:" + $k) -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($k, $prevEnv[$k], 'Process')
        }
    }
}

# ── Start frontend ──
$frontendProcess = $null
if (-not $BackendOnly) {
    $frontendCmd = if ($useProd) { 'start' } else { 'run dev' }
    step "Starting frontend on :$FrontendPort (npm $frontendCmd)"
    Write-Host "       logs: $FrontendLog"

    $env:NEXT_PUBLIC_BACKEND_URL = $BackendUrl
    $env:PORT                    = $FrontendPort

    $frontendProcess = Start-Process `
        -FilePath $NpmCmd `
        -ArgumentList @('run', $(if ($useProd) { 'start' } else { 'dev' })) `
        -WorkingDirectory $FrontendDir `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $FrontendLog `
        -RedirectStandardError  "$FrontendLog.err"
}

# ── Main wait loop ──
try {
    if ($backendProcess) {
        # Pass backend.log.err as the LogPath so the heartbeat surfaces the
        # latest gemlite/transformer/text-encoder loader stage instead of
        # just "still waiting...".
        $status = Wait-ForPort -Port $BackendPort -Name 'Backend' -ProcessId $backendProcess.Id -MaxSeconds $BackendReadyTimeout -LogPath "$BackendLog.err"
        switch ($status) {
            0 { }
            1 {
                err "Backend exited before port $BackendPort came up. Last lines of ${BackendLog}:"
                Write-Host "------------------------------------------------------------------------"
                if (Test-Path $BackendLog) { Get-Content $BackendLog -Tail 40 | Out-Host }
                if (Test-Path "$BackendLog.err") { Get-Content "$BackendLog.err" -Tail 40 | Out-Host }
                Write-Host "------------------------------------------------------------------------"
                exit 1
            }
            2 {
                warn "Backend didn't respond on :$BackendPort after ${BackendReadyTimeout}s -- check $LogDir\"
            }
        }
    }

    if ($frontendProcess) {
        # Frontend slow-start is non-fatal: next dev can be sluggish first-paint.
        $null = Wait-ForPort -Port $FrontendPort -Name 'Frontend' -ProcessId $frontendProcess.Id -MaxSeconds 120 -LogPath "$FrontendLog.err"
    }

    Write-Host ""
    Write-Host "========================================================================"
    Write-Host ""
    if ($frontendProcess) {
        Write-Host "  Frontend (open in your browser):"
        Write-Host "    http://localhost:$FrontendPort/"
        Write-Host ""
    }
    if ($backendProcess) {
        Write-Host "  Backend API:"
        Write-Host "    http://localhost:$BackendPort/              root"
        Write-Host "    http://localhost:$BackendPort/backends      available arms + GPU probe"
        Write-Host "    http://localhost:$BackendPort/docs          OpenAPI UI"
        Write-Host ""
    }
    Write-Host "  Logs: $LogDir\"
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "  Ctrl+C to stop."
    Write-Host ""

    # Wait on whichever processes we started; exit cleanly if either dies.
    $procIds = @()
    if ($backendProcess)  { $procIds += $backendProcess.Id }
    if ($frontendProcess) { $procIds += $frontendProcess.Id }
    if ($procIds.Count -gt 0) {
        Wait-Process -Id $procIds -ErrorAction SilentlyContinue
    }
}
finally {
    if ($backendProcess)  { info "Stopping backend (pid=$($backendProcess.Id))..."; Stop-ProcessTree -ProcessId $backendProcess.Id }
    if ($frontendProcess) { info "Stopping frontend (pid=$($frontendProcess.Id))..."; Stop-ProcessTree -ProcessId $frontendProcess.Id }
    # Belt-and-braces: next dev reparents next-server; sweep listening ports too.
    Start-Sleep -Seconds 1
    if ($backendProcess)  { Stop-PortListeners -Port $BackendPort }
    if ($frontendProcess) { Stop-PortListeners -Port $FrontendPort }
}
