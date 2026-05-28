# Send a /generate request to an already-running studio (PowerShell parallel
# to send_request.sh). Same flag surface; each call is a thin HTTP POST to
# the backend so weights stay resident across renders.
#
# Usage:
#   .\scripts\send_request.ps1 -Prompt "a tiny bonsai tree"
#   .\scripts\send_request.ps1 -p "..." -Size 1024x1024 -Seed 42 -Steps 8
#   $env:BACKEND_PORT = '8800'; .\scripts\send_request.ps1 -p "..."

[CmdletBinding()]
param(
    [Alias('p')]
    [Parameter(Mandatory)]
    [string]$Prompt,

    [int]$Seed,
    [int]$Steps  = 4,
    [string]$Size = '512x512',
    [string]$Output,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'

$DemoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Venv $DemoDir

$BackendPort = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { '8000' }
$BackendHost = if ($env:BACKEND_HOST) { $env:BACKEND_HOST } else { '127.0.0.1' }
$HostUrl     = "http://${BackendHost}:${BackendPort}"

# ── Parse --size WxH ──
if ($Size -notmatch '^(\d+)x(\d+)$') {
    err "Invalid -Size: $Size (expected WxH like 512x512)"
    exit 1
}
$Width  = [int]$Matches[1]
$Height = [int]$Matches[2]

# ── Pick a positive 31-bit seed if the user didn't supply one ──
$venvPy = Get-VenvPython $DemoDir
if (-not $PSBoundParameters.ContainsKey('Seed')) {
    $Seed = [int](& $venvPy -c 'import secrets; print(secrets.randbits(31))')
}

# ── Probe /backends for the active arm ──
# Used for two things: (a) the output dir under outputs/<arm>/, and (b) the
# `backend` field in the JSON payload. Sending it explicitly is important
# because backend_gpu's GenerateRequest defaults to "bonsai-binary-gemlite",
# which on a ternary-only install would trigger a transformer reload from
# the unset BINARY_TRANSFORMER_PATH env var (-> /root/models/bonsai-binary).
$modelLabel  = $null
$activeArm   = $null
try {
    $backendsJson = Invoke-RestMethod -Uri "$HostUrl/backends" -TimeoutSec 5 -ErrorAction Stop
    # /backends shape: { kind, default_family, supported_families, ... }
    # Compose the canonical arm string as "<family>-<kind>".
    if ($backendsJson.default_family -and $backendsJson.kind) {
        $activeArm = "$($backendsJson.default_family)-$($backendsJson.kind)"
    } elseif ($backendsJson.default) {
        $activeArm = $backendsJson.default
    }
    $modelLabel = switch ($activeArm) {
        'bonsai-ternary-mlx'     { 'ternary-mlx' }
        'bonsai-ternary-gemlite' { 'ternary-gemlite' }
        'bonsai-binary-mlx'      { 'binary-mlx' }
        'bonsai-binary-gemlite'  { 'binary-gemlite' }
        default                  { $null }
    }
} catch {
    $modelLabel = $null
    $activeArm  = $null
}
if (-not $modelLabel) { $modelLabel = Default-Model }
if (-not $modelLabel) { $modelLabel = 'unknown' }

# ── Default output path ──
if (-not $Output) {
    $ts = (Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss')
    $Output = Join-Path $DemoDir "outputs\$modelLabel\image_${ts}_seed${Seed}.png"
}
$outDir = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# ── Build JSON payload ──
# Use PowerShell's ConvertTo-Json rather than string interpolation so the
# prompt is JSON-escaped correctly (quotes, backslashes, newlines).
$payloadObj = [ordered]@{
    prompt = $Prompt
    seed   = $Seed
    steps  = $Steps
    height = $Height
    width  = $Width
}
if ($activeArm) { $payloadObj['backend'] = $activeArm }
$payload = $payloadObj | ConvertTo-Json -Compress

step "POST $HostUrl/generate  ($modelLabel, ${Width}x${Height}, seed=$Seed)"
$t0 = Get-Date

# Invoke-WebRequest streams the response body straight to -OutFile, which is
# what we want for the binary PNG. Non-2xx responses throw on PS 5.1, so we
# capture in try/catch and read $_.Exception.Response for the error body.
try {
    $oldProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri "$HostUrl/generate" `
            -Method Post `
            -Headers @{ 'Content-Type' = 'application/json' } `
            -Body $payload `
            -OutFile $Output `
            -TimeoutSec 600 `
            -UseBasicParsing | Out-Null
    } finally {
        $ProgressPreference = $oldProgress
    }
} catch {
    $resp = $_.Exception.Response
    if ($resp) {
        $code = [int]$resp.StatusCode
        err "HTTP $code from $HostUrl/generate"
        try {
            $stream = $resp.GetResponseStream()
            $reader = New-Object IO.StreamReader($stream)
            $body   = $reader.ReadToEnd()
            if ($body) { Write-Host "  body: $($body.Substring(0, [Math]::Min(500, $body.Length)))" }
        } catch { }
    } else {
        err "request failed - is serve.ps1 running on $HostUrl? ($($_.Exception.Message))"
    }
    if (Test-Path $Output) { Remove-Item $Output -Force }
    exit 1
}

$wall = [int]((Get-Date) - $t0).TotalSeconds

$MetaDir = Join-Path $DemoDir "outputs\$modelLabel"
$MetaPath = Join-Path $MetaDir "generations.json"

try {
    if (-not (Test-Path $MetaDir)) {
        New-Item -ItemType Directory -Path $MetaDir -Force | Out-Null
    }

    try {
        $relOutput = Resolve-Path -LiteralPath $Output -ErrorAction Stop | ForEach-Object {
            $_.Path.Substring($DemoDir.Length).TrimStart('\')
        }
    } catch {
        $relOutput = $Output
    }

    $record = [pscustomobject]@{
        timestamp        = (Get-Date).ToUniversalTime().ToString("o")
        model            = $modelLabel
        prompt           = $Prompt
        seed             = $Seed
        width            = $Width
        height           = $Height
        steps            = $Steps
        duration_seconds = [math]::Round([double]$wall, 3)
        output           = $relOutput
    }

    $existing = @()
    if (Test-Path $MetaPath) {
        try {
            $raw = Get-Content -Path $MetaPath -Raw -Encoding UTF8
            if ($raw -and $raw.Trim()) {
                $parsed = $raw | ConvertFrom-Json
                $existing = @($parsed)
            }
        } catch {
            $existing = @()
        }
    }

    $updated = @($existing) + @($record)
    $json = ConvertTo-Json -InputObject @($updated) -Depth 10
    [System.IO.File]::WriteAllText($MetaPath, $json + "`r`n", [System.Text.Encoding]::UTF8)
} catch {
    Write-Warning "generations.json write failed: $_"
    $MetaPath = ""
}

Write-Host ""
Write-Host "  prompt: $Prompt"
Write-Host "  seed:   $Seed"
Write-Host "  size:   ${Width}x${Height}"
Write-Host "  wall:   ${wall}s"
Write-Host "  path:   $Output"
Write-Host "  meta:   $MetaPath"

if ($Open) {
    Invoke-Item $Output
}
