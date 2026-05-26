# Shared helpers for Bonsai Image demo PowerShell scripts.
# Dot-source this file: . "$PSScriptRoot\common.ps1"
#
# Parallel to scripts/common.sh -- same env knobs
# (BONSAI_PACKAGE_MIN_AGE_DAYS, BONSAI_DISABLE_HF_TRANSFER, ...) and the
# same helper names (`info`, `warn`, `err`, `step`, `download`,
# `Default-Model`, ...).

$ErrorActionPreference = 'Stop'

# ── Models ──
# Bonsai-Image-4B variants. The demo's two reference backends:
#   ternary-mlx       macOS Apple Silicon -- mlx packed 2-bit
#   ternary-gemlite   Linux CUDA          -- gemlite packed 2-bit
# Windows has no first-party inference path; gemlite/hqq/triton are Linux-only.
# Windows can still download either variant (HuggingFace works everywhere)
# and run them via WSL2.

# Pick the default model for the current platform.
#   Default-Model              → ternary on this platform
#   Default-Model ternary      → ternary on this platform
#   Default-Model binary       → binary on this platform
# Returns "" if the platform has no inference backend (e.g. native Windows).
function Default-Model {
    param([string]$Variant = 'ternary')
    if (Is-AppleSilicon) { return "$Variant-mlx" }
    if (Is-Linux)        { return "$Variant-gemlite" }
    # Native Windows: no in-process backend. We still need a label so
    # download_model.ps1 can fetch *something*; default to the gemlite arm
    # since that's what a WSL2 user would want, and it's also what a CUDA
    # GPU on Windows would target if Triton ever ships proper Windows wheels.
    if (Is-Windows)      { return "$Variant-gemlite" }
    return ''
}

# ── Package age cutoff ──
# Minimum age (in days) for any package version we install via uv or npm.
# Same defense rationale as common.sh -- a fresh supply-chain compromise
# hasn't had time to be caught and yanked yet. Set
# BONSAI_PACKAGE_MIN_AGE_DAYS=0 to disable.
function Get-PackageCutoffDate {
    $days = if ($env:BONSAI_PACKAGE_MIN_AGE_DAYS) { [int]$env:BONSAI_PACKAGE_MIN_AGE_DAYS } else { 7 }
    return (Get-Date).ToUniversalTime().AddDays(-$days).ToString('yyyy-MM-dd')
}

# ── Colored output ──
# Mirrors info/warn/err/step from common.sh. PowerShell's Write-Host honors
# -ForegroundColor in any host, including modern Windows Terminal.
function info { param([string]$Msg) Write-Host "[OK]   $Msg" -ForegroundColor Green }
function warn { param([string]$Msg) Write-Host "[WARN] $Msg" -ForegroundColor Yellow }
function err  { param([string]$Msg) Write-Host "[ERR]  $Msg" -ForegroundColor Red }
function step { param([string]$Msg) Write-Host "==>    $Msg" -ForegroundColor Cyan }

# ── download(url, dest) ──
# PowerShell ships Invoke-WebRequest everywhere; no need to probe for curl.
function Invoke-Download {
    param(
        [Parameter(Mandatory)][string]$Url,
        [Parameter(Mandatory)][string]$Dest
    )
    $oldProgress = $ProgressPreference
    # Invoke-WebRequest's default progress bar slows downloads by ~10x on
    # Windows PowerShell 5 -- flip it off for the duration of the call.
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
    } finally {
        $ProgressPreference = $oldProgress
    }
}

# ── Resolve demo root (parent of scripts/) ──
function Resolve-DemoDir {
    return (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

# ── venv paths (Windows layout: .venv\Scripts\) ──
function Get-VenvPython {
    param([string]$DemoDir)
    return (Join-Path $DemoDir '.venv\Scripts\python.exe')
}

function Get-VenvBin {
    param([string]$DemoDir)
    return (Join-Path $DemoDir '.venv\Scripts')
}

# ── Ensure .venv exists and python is callable ──
# The bash version `. activate`s the venv, but we don't need that here:
# every call invokes .venv\Scripts\python.exe directly, which is enough to
# pick up the venv's interpreter and site-packages without modifying $env:PATH.
function Assert-Venv {
    param([string]$DemoDir)
    $venvPy = Get-VenvPython $DemoDir
    if (-not (Test-Path $venvPy)) {
        err "Python venv not found at $venvPy. Run .\setup.ps1 first."
        exit 1
    }
}

# ── Platform checks ──
# Written to work in BOTH Windows PowerShell 5.1 (no $IsWindows/$IsLinux/$IsMacOS
# automatic variables) and PowerShell 7+. Falls back to OSVersion.Platform +
# $env:OS, which are defined everywhere.
function Is-Windows {
    if ($null -ne (Get-Variable -Name IsWindows -Scope Global -ErrorAction SilentlyContinue)) {
        return [bool]$global:IsWindows
    }
    return $env:OS -eq 'Windows_NT'
}

function Is-Linux {
    if ($null -ne (Get-Variable -Name IsLinux -Scope Global -ErrorAction SilentlyContinue)) {
        return [bool]$global:IsLinux
    }
    return $false
}

function Is-MacOS {
    if ($null -ne (Get-Variable -Name IsMacOS -Scope Global -ErrorAction SilentlyContinue)) {
        return [bool]$global:IsMacOS
    }
    return $false
}

function Is-AppleSilicon {
    if (-not (Is-MacOS)) { return $false }
    return [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq 'Arm64'
}

function Has-NvidiaGpu {
    # Same semantics as common.sh's has_nvidia_gpu: truthy if either the
    # runtime driver utility or the toolkit compiler is on PATH. nvidia-smi
    # alone is enough to confirm a usable GPU on Windows.
    return [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue) -or `
           [bool](Get-Command nvcc       -ErrorAction SilentlyContinue)
}
