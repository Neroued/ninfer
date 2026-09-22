# swap-serve.ps1 - A/B swap between the current (OLD) and compare (NEW) ninfer builds.
#
# Stops the running ninfer-serve, starts the NEW (compare) build on the SAME port,
# and verifies it actually comes up (process stays alive AND /v1/models returns 200).
#   - On SUCCESS: leaves the NEW build running.
#   - On FAILURE: kills the NEW build and restarts the OLD build (safe rollback).
#
# Usage:
#   .\swap-serve.ps1 [model] [concurrency]     # try the NEW build, roll back to OLD on failure
#   .\swap-serve.ps1 -Rollback                 # skip NEW; stop current and start the OLD build
#
#   model:       1..4 (same meaning as serve.ps1; default 1)
#   concurrency: 1..8 (default 1)
#
# Env overrides (same as serve.ps1): NINFER_MAX_CONTEXT, NINFER_SPEC, NINFER_MIN_FREE_GIB,
#   NINFER_SKIP_GPU_CHECK. Plus: NINFER_SWAP_TIMEOUT (readiness timeout seconds, default 240).
#
# Logs land in debug\swap-<new|old>.{out,err}.log.

param(
    [int]$Model = 1,
    [int]$Concurrency = 1,
    [switch]$Rollback
)
$ErrorActionPreference = "Stop"

# Resolve all runtime paths (build tree, models, DLLs, logs) portably.
# See win_paths.ps1: env override -> repo layout -> workspace-root fallback.
. (Join-Path $PSScriptRoot "win_paths.ps1")
# OLD = current known-good build (rollback baseline); NEW = candidate in ninfer-compare\build.
$OLD_BIN   = $NINFER_BIN
$NEW_BIN   = $NINFER_COMPARE_BIN
$BIND_HOST = "0.0.0.0"
$PORT      = 8080
$TIMEOUT   = if ($env:NINFER_SWAP_TIMEOUT) { [int]$env:NINFER_SWAP_TIMEOUT } else { 240 }

# The exe imports FFMPEG/libcurl DLLs even for text-only; put them on PATH.
if (Test-Path $FFMPEG_BIN) { $env:PATH = "$FFMPEG_BIN;$env:PATH" }
if (Test-Path $CURL_BIN)   { $env:PATH = "$CURL_BIN;$env:PATH" }

if (-not (Test-Path $OLD_BIN)) { Write-Error "missing OLD binary: $OLD_BIN"; exit 1 }
if (-not (Test-Path $NEW_BIN)) { Write-Error "missing NEW binary: $NEW_BIN (build ninfer-compare first)"; exit 1 }
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# --- model selection (mirrors serve.ps1) ---
$MAX_CONTEXT    = if ($env:NINFER_MAX_CONTEXT) { $env:NINFER_MAX_CONTEXT } else { "262144" }
$SPEC           = if ($env:NINFER_SPEC)        { $env:NINFER_SPEC }        else { "mtp3" }
$MIN_FREE_GIB   = if ($env:NINFER_MIN_FREE_GIB) { [int]$env:NINFER_MIN_FREE_GIB } else { 24 }
$MAX_GEN_TOKENS = 65536
$MODEL_ID       = "256k"

switch ($Model) {
    1 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_nvfp4.ninfer";    $VISION = 0; $DESC = "NVFP4 (text-only)" }
    2 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b.ninfer";          $VISION = 1; $DESC = "groupwise-int (vision)" }
    3 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_heretic.ninfer";  $VISION = 1; $DESC = "heretic (vision)" }
    4 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_nvfp4.ninfer";    $VISION = 1; $DESC = "NVFP4 + vision" }
    default { Write-Error "unknown model: $Model (use 1, 2, 3, or 4)"; exit 2 }
}
if (-not (Test-Path $ARTIFACT)) { Write-Error "missing artifact: $ARTIFACT"; exit 1 }

# --- speculative-decoding args (mirrors serve.ps1) ---
$SPEC_ARGS = @()
if ($SPEC -match "^mtp([1-5])$") {
    $SPEC_ARGS += @("--spec", "mtp", "--draft-tokens", $Matches[1], "--lm-head-draft")
}
elseif ($SPEC -match "^dflash([1-9]|1[0-5])$") {
    $SPEC_ARGS += @("--spec", "dflash", "--draft-tokens", $Matches[1])
}
elseif ($SPEC -ne "none" -and $SPEC -ne "mtp0") {
    Write-Error "invalid NINFER_SPEC: $SPEC (use none | mtp1..mtp5 | dflash1..dflash15)"; exit 2
}
if ($SPEC -like "dflash*" -and $VISION -eq 1) {
    Write-Error "NINFER_SPEC=$SPEC cannot be combined with vision model $Model"; exit 2
}

$NINFER_ARGS = @($ARTIFACT,
    "--host", $BIND_HOST,
    "--port", $PORT,
    "--model-id", $MODEL_ID,
    "--max-context", $MAX_CONTEXT,
    "--default-max-tokens", $MAX_GEN_TOKENS,
    "--kv-dtype", "int8",
    "--kv-capacity", "auto",
    "--max-concurrency", $Concurrency
) + $SPEC_ARGS
if ($VISION -eq 1) { $NINFER_ARGS += "--vision" }
# Drop any null/empty elements so Start-Process -ArgumentList never sees a null.
$NINFER_ARGS = @($NINFER_ARGS | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

# --- helpers ---
function Stop-RunningServer {
    Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
    for ($i = 0; $i -lt 30; $i++) {
        $inUse = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
        if (-not $inUse) { break }
        Start-Sleep -Milliseconds 500
    }
    Start-Sleep -Milliseconds 1500   # let the driver release VRAM
}

function Test-GpuFree {
    if ($env:NINFER_SKIP_GPU_CHECK -eq "1") { return $true }
    $smi = nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { Write-Warning "nvidia-smi failed; skipping GPU check"; return $true }
    $parts = ($smi -split ",")[0..1]
    $free  = [math]::Floor(([int]$parts[0].Trim() - [int]$parts[1].Trim()) / 1024)
    if ($free -lt $MIN_FREE_GIB) {
        Write-Warning "GPU not free enough: ${free} GiB free (need ${MIN_FREE_GIB})."
        return $false
    }
    Write-Host "GPU check: ${free} GiB free (need ${MIN_FREE_GIB})"
    return $true
}

function Start-Server {
    param([string]$Bin, [string]$Tag)
    $logOut = Join-Path $LOG_DIR "swap-$Tag.out.log"
    $logErr = Join-Path $LOG_DIR "swap-$Tag.err.log"
    if (Test-Path $logOut) { Remove-Item $logOut -Force }
    if (Test-Path $logErr) { Remove-Item $logErr -Force }
    $proc = Start-Process -FilePath $Bin -ArgumentList $NINFER_ARGS -PassThru `
        -RedirectStandardOutput $logOut -RedirectStandardError $logErr -WindowStyle Hidden
    return @{ Process = $proc; LogErr = $logErr; LogOut = $logOut }
}

function Test-Ready {
    param($Info, [int]$TimeoutSec)
    $proc     = $Info.Process
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            return @{ Ok = $false; Reason = "process exited (code $($proc.ExitCode))" }
        }
        try {
            # Use 127.0.0.1 explicitly: the server binds 0.0.0.0 (IPv4) and on Windows
            # 'localhost' can resolve to ::1 (IPv6) first, which never connects.
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$PORT/v1/models" -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return @{ Ok = $true } }
        } catch { }
        Start-Sleep -Milliseconds 1000
    }
    return @{ Ok = $false; Reason = "readiness timeout after ${TimeoutSec}s" }
}

function Stop-Server {
    param($Info)
    if ($null -ne $Info.Process -and -not $Info.Process.HasExited) {
        Stop-Process -Id $Info.Process.Id -Force -ErrorAction SilentlyContinue
    }
}

function Show-Tail {
    param([string]$Path, [string]$Indent)
    if (Test-Path $Path) {
        Get-Content $Path -Tail 25 | ForEach-Object { Write-Host "$Indent$_" }
    }
}

function Invoke-StartAndVerify {
    param([string]$Bin, [string]$Tag, [string]$Label)
    Write-Host "Starting $Label build..."
    $info = Start-Server -Bin $Bin -Tag $Tag
    $ready = Test-Ready -Info $info -TimeoutSec $TIMEOUT
    if ($ready.Ok) {
        Write-Host "[OK] $Label build is serving on http://localhost:$PORT (model id: $MODEL_ID)."
        Write-Host "     logs: $($info.LogErr)"
        return $true
    }
    Write-Host "[FAIL] $Label build did not come up: $($ready.Reason)"
    Write-Host "       last stderr:"
    Show-Tail -Path $info.LogErr -Indent "       "
    Stop-Server -Info $info
    return $false
}

# --- main ---
Write-Host "=================================================="
Write-Host "   NInfer A/B swap  (model ${Model}: $DESC)"
Write-Host "   OLD: $OLD_BIN"
Write-Host "   NEW: $NEW_BIN"
Write-Host "   port: $PORT   spec: $SPEC   max-context: $MAX_CONTEXT"
Write-Host "=================================================="

if ($Rollback) {
    Write-Host "`nRollback requested: stopping current, starting OLD build."
    Stop-RunningServer
    Test-GpuFree | Out-Null
    if (Invoke-StartAndVerify -Bin $OLD_BIN -Tag "old" -Label "OLD") { exit 0 }
    Write-Host "[CRITICAL] OLD build failed to come up. Nothing is serving on port $PORT."
    exit 4
}

Stop-RunningServer
if (-not (Test-GpuFree)) {
    Write-Error "GPU not free enough to start the NEW build. Bypass with NINFER_SKIP_GPU_CHECK=1."
    exit 3
}

Write-Host ""
if (Invoke-StartAndVerify -Bin $NEW_BIN -Tag "new" -Label "NEW") {
    Write-Host "`nSwap complete: NEW (compare) build is now serving. The OLD build was stopped."
    Write-Host "To roll back to the OLD build at any time:  .\swap-serve.ps1 -Rollback"
    exit 0
}

Write-Host "`nNEW build failed - rolling back to the OLD build..."
Stop-RunningServer
Test-GpuFree | Out-Null
if (Invoke-StartAndVerify -Bin $OLD_BIN -Tag "old" -Label "OLD") {
    Write-Host "`nRollback complete: OLD build is serving again. Investigate debug\swap-new.err.log."
    exit 0
}
Write-Host "[CRITICAL] OLD build also failed to come up. Nothing is serving on port $PORT."
exit 4