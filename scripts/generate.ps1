# Thin wrapper around scripts/generate.py (PowerShell parallel to generate.sh).
#
# Native Windows runs the same code path as Linux GPU (gemlite + hqq +
# triton, with triton-windows providing the `triton` module). generate.py
# refuses non-darwin in-process runs without --force-gpu-run, so this
# wrapper injects the flag automatically on Windows.
#
# Usage:
#   .\scripts\generate.ps1 -p "a tiny bonsai tree"
#   .\scripts\generate.ps1 -p "a red cube" --steps 8 --seed 42
#   .\scripts\generate.ps1 -p "..." --size 1024x1024

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = 'Stop'

$DemoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Venv $DemoDir

# Force UTF-8 stdout so generate.py's banner (which uses non-ASCII glyphs
# like the hourglass + lightning emoji and the multiplication sign) doesn't
# crash on Windows cp1252.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

$venvPy = Get-VenvPython $DemoDir
$genPy  = Join-Path $DemoDir 'scripts\generate.py'

# On Windows, opt into the in-process GPU run automatically -- generate.py
# guards this behind --force-gpu-run on non-darwin so users default to
# serve.sh, but if you're calling generate.ps1 you've already chosen the
# one-shot path. Don't add it twice if the caller already passed it.
$genArgs = @()
if (Is-Windows -and -not ($ForwardArgs -contains '--force-gpu-run')) {
    $genArgs += '--force-gpu-run'
}
if ($ForwardArgs) { $genArgs += $ForwardArgs }

& $venvPy $genPy @genArgs
exit $LASTEXITCODE
