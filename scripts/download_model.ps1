# Download a model for the Bonsai Image demo (PowerShell parallel to
# download_model.sh).
#
# Usage:
#   .\scripts\download_model.ps1                     # ternary (default), gemlite on Windows
#   .\scripts\download_model.ps1 ternary             # same -- explicit
#   .\scripts\download_model.ps1 binary              # binary 1-bit
#   .\scripts\download_model.ps1 -Model binary-gemlite
#
# Short form (ternary | binary) picks the gemlite arm on Windows since
# that's the only set of weights you'd later run via WSL2; mlx weights are
# Apple-Silicon-only and pointless to download here. Long form
# (ternary-mlx | binary-gemlite | ...) overrides explicitly.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Variant,
    [Alias('m')]
    [string]$Model
)

$ErrorActionPreference = 'Stop'

$DemoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Venv $DemoDir

function Show-Usage {
@"
Usage: download_model.ps1 [<variant>] [-Model <full-name>]

  Short form (picks backend automatically per platform):
    ternary        -> ternary-gemlite on Windows / Linux
    binary         -> binary-gemlite  on Windows / Linux

  Long form (explicit backend):
    ternary-mlx       prism-ml/bonsai-image-ternary-4B-mlx-2bit
    ternary-gemlite   prism-ml/bonsai-image-ternary-4B-gemlite-2bit
    binary-mlx        prism-ml/bonsai-image-binary-4B-mlx-1bit
    binary-gemlite    prism-ml/bonsai-image-binary-4B-gemlite-1bit

If nothing is passed, ternary for this platform is picked.
"@
}

# Either bare positional ($Variant) or -Model can carry the choice.
# Bare positional accepts the short or long form.
if (-not $Model -and $Variant) {
    $Model = $Variant
}
if (-not $Model) {
    $Model = Default-Model   # ternary on this platform
}

# Short form → expand to <variant>-<backend> for this platform.
if ($Model -in 'ternary','binary') {
    $full = Default-Model $Model
    if (-not $full) {
        err "Couldn't pick a backend for this platform."
        err "Pass -Model $Model-mlx or -Model $Model-gemlite explicitly."
        exit 1
    }
    $Model = $full
}

if ($Model -notin 'ternary-mlx','binary-mlx','ternary-gemlite','binary-gemlite') {
    err "Invalid model: $Model (must be ternary-mlx | ternary-gemlite | binary-mlx | binary-gemlite)"
    Show-Usage
    exit 1
}

# Derive bits + HF repo + local dir from the <variant>-<backend> name.
$parts   = $Model.Split('-', 2)
$variant = $parts[0]              # ternary | binary
$backend = $parts[1]              # mlx | gemlite
$bits    = if ($variant -eq 'binary') { 1 } else { 2 }
$savedDir = Join-Path $DemoDir "models\bonsai-image-4B-$variant-$backend"
$display  = "Bonsai-Image-4B ($variant $backend-${bits}bit)"
$hfRepo   = "prism-ml/bonsai-image-$variant-4B-$backend-${bits}bit"

# Always call snapshot_download. It is idempotent: fresh download, resume
# of an interrupted partial, or no-op on a complete dir all flow through
# the same code path. No need for us to detect .incomplete files or
# anything else -- HF handles it.
step "Fetching $display into $savedDir ..."
Write-Host "  (HuggingFace snapshot_download: fresh / resume / verify, whichever applies)"

# HF_HUB_ENABLE_HF_TRANSFER=1 switches the per-file downloader to
# hf_transfer (Rust, parallel HTTP range requests). Bumps a stalled-link
# 10-20 MB/s baseline up to saturating residential gigabit on typical
# files. Set BONSAI_DISABLE_HF_TRANSFER=1 to fall back to the python
# requests backend.
if ($env:BONSAI_DISABLE_HF_TRANSFER -ne '1') {
    $env:HF_HUB_ENABLE_HF_TRANSFER = '1'
}

# Pass repo id via env var instead of string-interpolating into the Python
# source -- keeps the user-controlled value out of the parsed code path.
# Same effect as the bash version's `env VAR=... python -c ...`, but immune
# to characters that would otherwise need escaping.
$env:BONSAI_HF_REPO      = $hfRepo
$env:BONSAI_HF_LOCAL_DIR = $savedDir

$pyCode = @"
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ['BONSAI_HF_REPO'],
    local_dir=os.environ['BONSAI_HF_LOCAL_DIR'],
    max_workers=16,
)
"@

$venvPy = Get-VenvPython $DemoDir
& $venvPy -c $pyCode
$rc = $LASTEXITCODE

# Don't leak the token into the parent shell after we return.
Remove-Item Env:BONSAI_HF_TOKEN_PASS -ErrorAction SilentlyContinue
Remove-Item Env:BONSAI_HF_REPO       -ErrorAction SilentlyContinue
Remove-Item Env:BONSAI_HF_LOCAL_DIR  -ErrorAction SilentlyContinue

if ($rc -ne 0) {
    err "snapshot_download failed (exit $rc)."
    exit $rc
}

info "Model saved to $savedDir"
