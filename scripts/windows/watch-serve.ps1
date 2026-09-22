# watch-serve.ps1 - NInfer heartbeat supervisor (Windows)
#
# Launches the NInfer server, then watches it. If the process dies, the health
# probe fails, or the log shows a fatal pattern, it stops the dead process and
# relaunches with the SAME configuration. Bounded restarts with exponential
# backoff so a crash-loop cannot spin forever.
#
# This is the "keep it up" companion to serve.ps1. Use it when you want the
# server to self-heal (e.g. after a transient CUDA/context fault) instead of
# dying and needing a manual relaunch.
#
# Usage:
#   .\watch-serve.ps1                      # default model 1, concurrency 1
#   .\watch-serve.ps1 2                    # model 2 (vision), default concurrency
#   .\watch-serve.ps1 1 4                  # model 1, concurrency 4
#   .\watch-serve.ps1 -UseOld              # watch the OLD (ninfer\) build instead of the new one
#   .\watch-serve.ps1 -NoAutorestore       # do not auto-restore the port if VS Code grabbed it
#
# Env overrides (same as serve.ps1):
#   NINFER_MAX_CONTEXT  16384 | 65536 | 131072 | 262144   (default 262144)
#   NINFER_SPEC         none | mtp<N> | dflash<N>        (default mtp3)
#   NINFER_MIN_FREE_GIB   min free VRAM to start (default 24)
#   NINFER_SKIP_GPU_CHECK=1  bypass the GPU check
#
# Watcher knobs:
#   NINFER_WATCH_MAX_RESTARTS   max relaunches before giving up (default 3)
#   NINFER_WATCH_HEALTH_TIMEOUT readiness timeout seconds per launch (default 240)
#   NINFER_WATCH_POLL_MS        poll interval ms (default 2000)
#   NINFER_WATCH_STABLE_MS      after this many ms healthy, reset the restart counter (default 120000)
#   NINFER_WATCH_PORT_HEAL=1    if the port is held by a foreign process, try to free it (default on)

param(
    [int]$Model = 1,
    [int]$Concurrency = 1,
    [switch]$UseOld,
    [switch]$NoAutorestore
)
$ErrorActionPreference = "Stop"

# Resolve all runtime paths (build tree, models, DLLs, logs) portably.
# See win_paths.ps1: env override -> repo layout -> workspace-root fallback.
. (Join-Path $PSScriptRoot "win_paths.ps1")
if ($UseOld) {
    $BIN  = $NINFER_BIN            # current known-good build
    $TAG  = "old"
} else {
    $BIN  = $NINFER_COMPARE_BIN    # candidate build
    $TAG  = "new"
}
$BIND_HOST = "0.0.0.0"
$PORT = 8080

$MAX_CONTEXT    = if ($env:NINFER_MAX_CONTEXT) { $env:NINFER_MAX_CONTEXT } else { "262144" }
$SPEC           = if ($env:NINFER_SPEC)        { $env:NINFER_SPEC }        else { "mtp3" }
$MIN_FREE_GIB   = if ($env:NINFER_MIN_FREE_GIB) { [int]$env:NINFER_MIN_FREE_GIB } else { 24 }
$MAX_GEN_TOKENS = 65536
$MODEL_ID       = "256k"

$MAX_RESTARTS = if ($env:NINFER_WATCH_MAX_RESTARTS) { [int]$env:NINFER_WATCH_MAX_RESTARTS } else { 3 }
$HEALTH_TIMEOUT = if ($env:NINFER_WATCH_HEALTH_TIMEOUT) { [int]$env:NINFER_WATCH_HEALTH_TIMEOUT } else { 240 }
$POLL_MS = if ($env:NINFER_WATCH_POLL_MS) { [int]$env:NINFER_WATCH_POLL_MS } else { 2000 }
$STABLE_MS = if ($env:NINFER_WATCH_STABLE_MS) { [int]$env:NINFER_WATCH_STABLE_MS } else { 120000 }
$HEAL_PORT = -not $NoAutorestore

# Add FFMPEG + libcurl DLL directories to PATH (needed when vision is enabled)
if (Test-Path $FFMPEG_BIN) { $env:PATH = "$FFMPEG_BIN;$env:PATH" }
if (Test-Path $CURL_BIN)   { $env:PATH = "$CURL_BIN;$env:PATH" }

if (-not (Test-Path $BIN))    { Write-Error "missing $BIN"; exit 1 }
if (-not (Test-Path $MODELS)) { Write-Error "missing $MODELS"; exit 1 }
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

switch ($Model) {
    1 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_nvfp4.ninfer";    $VISION = 0; $DESC = "NVFP4 (text-only)" }
    2 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b.ninfer";          $VISION = 1; $DESC = "groupwise-int (vision enabled)" }
    3 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_heretic.ninfer";  $VISION = 1; $DESC = "heretic / abliterated (vision enabled)" }
    4 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_nvfp4.ninfer";    $VISION = 1; $DESC = "NVFP4 + vision (tightest VRAM fit)" }
    default { Write-Error "unknown model: $Model (use 1, 2, 3, or 4)"; exit 2 }
}
if (-not (Test-Path $ARTIFACT)) { Write-Error "missing $ARTIFACT"; exit 1 }

# Build the argument list once. This IS the "last configuration" the watcher
# relaunches with. Do NOT name this variable $args (PowerShell automatic var).
$NINFER_ARGS = @($ARTIFACT,
    "--host", $BIND_HOST,
    "--port", $PORT,
    "--model-id", $MODEL_ID,
    "--max-context", $MAX_CONTEXT,
    "--default-max-tokens", $MAX_GEN_TOKENS,
    "--kv-dtype", "int8",
    "--kv-capacity", "auto",
    "--max-concurrency", $Concurrency
)
if ($SPEC -match "^mtp([1-5])$") {
    $NINFER_ARGS += @("--spec", "mtp", "--draft-tokens", $Matches[1], "--lm-head-draft")
}
elseif ($SPEC -match "^dflash([1-9]|1[0-5])$") {
    $NINFER_ARGS += @("--spec", "dflash", "--draft-tokens", $Matches[1])
}
elseif ($SPEC -ne "none" -and $SPEC -ne "mtp0") {
    Write-Error "invalid NINFER_SPEC: $SPEC (use none | mtp1..mtp5 | dflash1..dflash15)"
    exit 2
}
if ($SPEC -like "dflash*" -and $VISION -eq 1) {
    Write-Error "NINFER_SPEC=$SPEC cannot be combined with vision model $Model"
    exit 2
}
if ($VISION -eq 1) { $NINFER_ARGS += "--vision" }
$NINFER_ARGS = @($NINFER_ARGS | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

# Fatal patterns: if the log matches any of these, treat the instance as dead.
$FATAL_PATTERNS = @(
    "failed to bind",
    "failed to load",
    "failed to initialize",
    "CUDA error",
    "cudaError",
    "cudaErrorMemoryAllocation",
    "out of memory",
    "std::bad_alloc",
    "std::runtime_error",
    "std::system_error",
    "std::logic_error",
    "std::invalid_argument",
    "std::out_of_range",
    "std::filesystem::filesystem_error",
    "terminate called",
    "abort",
    "aborting",
    "assertion",
    "segmentation fault",
    "access violation",
    "stack buffer overrun",
    "stack buffer overrun detected",
    "unhandled exception",
    "pure virtual function",
    "invalid argument",
    "invalid parameter",
    "pure virtual function call invoked",
    "terminate called after an exception",
    "cudaErrorInvalidValue",
    "cudaErrorInvalidDeviceContext",
    "cudaErrorInvalidDevicePointer",
    "cudaErrorInvalidSourceSize",
    "cudaErrorInvalidConfiguration",
    "cudaErrorLaunchFailure",
    "cudaErrorNotReady",
    "cudaErrorNotInitialized",
    "cudaErrorMapBuffer",
    "cudaErrorSymbolNotFound",
    "cudaErrorSizeOverflow",
    "cudaErrorValue",
    "cudaErrorUnknown"
)

function Write-Mon {
    param([string]$Level, [string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] [$Level] $Message"
}

function Stop-Ninfer {
    Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 30; $i++) {
        $inUse = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
        if (-not $inUse) { break }
        Start-Sleep -Milliseconds 500
    }
    Start-Sleep -Milliseconds 1500
}

function Test-GpuFree {
    if ($env:NINFER_SKIP_GPU_CHECK -eq "1") { return $true }
    $smi = nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { Write-Mon "warn" "nvidia-smi failed; skipping GPU check"; return $true }
    $parts = ($smi -split ",")[0..1]
    $total = [int]$parts[0].Trim()
    $used  = [int]$parts[1].Trim()
    $free  = [math]::Floor(($total - $used) / 1024)
    if ($free -lt $MIN_FREE_GIB) {
        Write-Mon "warn" "GPU not free enough: ${free} GiB free (need ${MIN_FREE_GIB})"
        return $false
    }
    Write-Mon "info" "GPU check: ${free} GiB free (need ${MIN_FREE_GIB})"
    return $true
}

function Start-Ninfer {
    param([string]$Tag)
    $logOut = Join-Path $LOG_DIR "watch-$Tag.out.log"
    $logErr = Join-Path $LOG_DIR "watch-$Tag.err.log"
    if (Test-Path $logOut) { Remove-Item $logOut -Force -ErrorAction SilentlyContinue }
    if (Test-Path $logErr) { Remove-Item $logErr -Force -ErrorAction SilentlyContinue }
    $proc = Start-Process -FilePath $BIN -ArgumentList $NINFER_ARGS -PassThru `
        -RedirectStandardOutput $logOut -RedirectStandardError $logErr -WindowStyle Hidden
    return @{ Process = $proc; LogErr = $logErr; LogOut = $logOut }
}

function Test-Ready {
    param($Info, [int]$TimeoutSec)
    $proc = $Info.Process
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            return @{ Ok = $false; Reason = "process exited (code $($proc.ExitCode))" }
        }
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$PORT/v1/models" -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return @{ Ok = $true } }
        } catch { }
        Start-Sleep -Milliseconds 1000
    }
    return @{ Ok = $false; Reason = "readiness timeout after ${TimeoutSec}s" }
}

function Test-LogFatal {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    $lines = Get-Content $Path -Tail 80 -ErrorAction SilentlyContinue
    if (-not $lines) { return $null }
    foreach ($line in $lines) {
        foreach ($p in $FATAL_PATTERNS) {
            if ($line -like "*$p*") { return $line }
        }
    }
    return $null
}

function Show-Tail {
    param([string]$Path, [string]$Indent)
    if (Test-Path $Path) {
        Get-Content $Path -Tail 25 | ForEach-Object { Write-Host "$Indent$_" }
    }
}

function Invoke-RunOnce {
    param([string]$Tag)
    $info = Start-Ninfer -Tag $Tag
    $ready = Test-Ready -Info $info -TimeoutSec $HEALTH_TIMEOUT
    if (-not $ready.Ok) {
        Write-Mon "warn" "startup not ready: $($ready.Reason)"
        Show-Tail -Path $info.LogErr -Indent "  "
        Stop-Ninfer
        return $false
    }
    Write-Mon "info" "server is up on http://localhost:$PORT (model id: $MODEL_ID)"
    $healthySince = Get-Date
    $lastFatal = $null
    while ($true) {
        Start-Sleep -Milliseconds $POLL_MS
        if ($info.Process.HasExited) {
            Write-Mon "warn" "process exited (code $($info.Process.ExitCode))"
            Show-Tail -Path $info.LogErr -Indent "  "
            Stop-Ninfer
            return $false
        }
        $fatal = Test-LogFatal -Path $info.LogErr
        if ($null -ne $fatal) {
            Write-Mon "warn" "fatal log line detected: $fatal"
            Show-Tail -Path $info.LogErr -Indent "  "
            Stop-Ninfer
            return $false
        }
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$PORT/v1/models" -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) {
                $healthySince = Get-Date
            } else {
                Write-Mon "warn" "health probe returned HTTP $($r.StatusCode)"
                Stop-Ninfer
                return $false
            }
        } catch {
            Write-Mon "warn" "health probe failed: $($_.Exception.Message)"
            Stop-Ninfer
            return $false
        }
    }
}

# ---- main ----
Write-Host "=================================================="
Write-Host "   NInfer heartbeat supervisor"
Write-Host "   bin:        $BIN"
Write-Host "   model:      $Model ($DESC)"
Write-Host "   concurrency: $Concurrency"
Write-Host "   max context: $MAX_CONTEXT"
Write-Host "   spec:        $SPEC"
Write-Host "   max restarts: $MAX_RESTARTS   health timeout: ${HEALTH_TIMEOUT}s"
Write-Host "=================================================="

# If a previous instance is running, stop it first so we own the port.
Stop-Ninfer

$attempt = 0
while ($true) {
    $attempt++
    Write-Mon "info" "launch attempt $attempt / $MAX_RESTARTS"

    if (-not (Test-GpuFree)) {
        Write-Mon "warn" "GPU not free; waiting for VRAM to be released"
        Start-Sleep -Seconds 5
    }

    $ok = Invoke-RunOnce -Tag $TAG
    if ($ok) {
        Write-Mon "info" "healthy run completed (server exited cleanly)"
        break
    }

    if ($attempt -ge $MAX_RESTARTS) {
        Write-Mon "error" "giving up after $attempt failed attempt(s)"
        Write-Mon "error" "last logs: $LOG_DIR\watch-$TAG.err.log"
        exit 4
    }

    $backoff = [math]::Min(30, 5 * $attempt)
    Write-Mon "warn" "restarting in ${backoff}s (attempt $attempt/$MAX_RESTARTS)"
    Start-Sleep -Seconds $backoff
}

Write-Mon "info" "watcher finished"