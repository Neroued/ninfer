# serve.ps1 - NInfer server launcher (Windows native)
#
# 1:1 port of the WSL launcher (ninfer/serve.sh on the WSL side):
#   .\serve.ps1 <model> [concurrency]
#
#   model:       1 = nvfp4         (text-only, fast W4A4 prefill)
#                2 = groupwise-int (vision enabled)
#                3 = heretic       (abliterated, vision enabled)
#                4 = nvfp4+vision  (fast W4A4 prefill, tightest VRAM fit)
#   concurrency: 1..8 (default 1)
#
# Capability matrix (RTX 5090, 32 GiB, verified 2026-08-19):
#   nvfp4:     full 256k context, NO vision (18.98 GiB weights leave no room)
#   groupwise: full 256k context + vision
#   heretic:   full 256k context + vision
#   nvfp4+vis: vision enabled; tightest fit (~19.3 GiB weights). If the auto KV
#              pool can't hold one max-context sequence at startup, lower
#              NINFER_MAX_CONTEXT (and/or NINFER_MIN_FREE_GIB) to fit your card.
#
# Env overrides:
#   NINFER_MAX_CONTEXT  16384 | 65536 | 131072 | 262144   (default 131072)
#   NINFER_SPEC         none | mtp<N> | dflash<N>          (default mtp3)
#     e.g. $env:NINFER_MAX_CONTEXT=262144; $env:NINFER_SPEC=none; .\serve.ps1 1
#          -> full 256K context, no MTP (~71 tok/s)
#   NINFER_MIN_FREE_GIB   min free VRAM to start (default 24; covers 19.7 GiB
#                         weights + KV pool + 1 GiB headroom at 128K/256K)
#   NINFER_SKIP_GPU_CHECK=1  bypass the check (e.g. you know what you're doing)
#
# Context + speculative decoding defaults: 256K context + MTP3.
# For faster throughput at shorter context, override:
#   $env:NINFER_MAX_CONTEXT=131072; $env:NINFER_SPEC=mtp3; .\serve.ps1 1
#   -> 128K context, ~117 tok/s

param(
    [int]$Model = 1,
    [int]$Concurrency = 1
)
$ErrorActionPreference = "Stop"

# Resolve all runtime paths (build tree, models, DLLs, logs) portably.
# See win_paths.ps1: env override -> repo layout -> workspace-root fallback.
. (Join-Path $PSScriptRoot "win_paths.ps1")
$BIN = $NINFER_BIN

# Add FFMPEG + libcurl DLL directories to PATH (needed when vision is enabled)
if (Test-Path $FFMPEG_BIN) { $env:PATH = "$FFMPEG_BIN;$env:PATH" }
if (Test-Path $CURL_BIN)   { $env:PATH = "$CURL_BIN;$env:PATH" }
$BIND_HOST  = "0.0.0.0"
$PORT       = 8080
$MAX_CONTEXT    = if ($env:NINFER_MAX_CONTEXT) { $env:NINFER_MAX_CONTEXT } else { "262144" }
$SPEC           = if ($env:NINFER_SPEC)        { $env:NINFER_SPEC }        else { "mtp3" }
$MIN_FREE_GIB   = if ($env:NINFER_MIN_FREE_GIB) { [int]$env:NINFER_MIN_FREE_GIB } else { 24 }
$MAX_GEN_TOKENS = 65536      # default max output tokens (high-thinking model; engine clamps to remaining context)
$MODEL_ID       = "256k"     # public alias clients use in the "model" field

switch ($Model) {
    1 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_nvfp4.ninfer";    $VISION = 0; $DESC = "NVFP4 (text-only, fast W4A4 prefill)" }
    2 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b.ninfer";          $VISION = 1; $DESC = "groupwise-int (vision enabled)" }
    3 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_heretic.ninfer";  $VISION = 1; $DESC = "heretic / abliterated (vision enabled)" }
    4 { $ARTIFACT = Join-Path $MODELS "qwen3_8_27b_nvfp4.ninfer";    $VISION = 1; $DESC = "NVFP4 + vision (fast W4A4 prefill, tightest VRAM fit)" }
    default { Write-Error "unknown model: $Model (use 1, 2, 3, or 4)"; exit 2 }
}

if (-not (Test-Path $BIN))      { Write-Error "missing $BIN - build first (see windows-port-tasks.md)"; exit 1 }
if (-not (Test-Path $ARTIFACT)) { Write-Error "missing $ARTIFACT"; exit 1 }

# GPU co-residency guard: the engine's startup reservation is all-or-nothing
# (weights + KV pool + 1 GiB headroom), so if anything else is holding VRAM,
# startup fails with a confusing "only N bytes available after weights" error.
# Fail early with a clear message instead.
if ($env:NINFER_SKIP_GPU_CHECK -ne "1") {
    $smi = nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed" }
    $parts = ($smi -split ",")[0..1]
    $total = [int]$parts[0].Trim()
    $used  = [int]$parts[1].Trim()
    $free  = [math]::Floor(($total - $used) / 1024)
    if ($free -lt $MIN_FREE_GIB) {
        Write-Error "GPU not free enough to start: ${free} GiB free (need ${MIN_FREE_GIB} GiB).`n  Another process is holding VRAM.`n  Stop it and retry, or bypass with NINFER_SKIP_GPU_CHECK=1."
        exit 3
    }
    Write-Host "GPU check: ${free} GiB free (need ${MIN_FREE_GIB})"
}

# Stop any running server, then wait for the port to be free.
Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
for ($i = 0; $i -lt 20; $i++) {
    $inUse = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
    if (-not $inUse) { break }
    Start-Sleep -Milliseconds 500
}

# Build speculative-decoding flags from $SPEC (none | mtp<N> | dflash<N>).
# Engine constraints (src/product/speculative_options.h): MTP draft in [1,5],
# DFlash draft in [1,15]. mtp0 == no speculation, so map it to none.
$SPEC_ARGS = @()
if ($SPEC -match "^mtp([1-5])$") {
    $SPEC_ARGS += @("--spec", "mtp", "--draft-tokens", $Matches[1], "--lm-head-draft")
}
elseif ($SPEC -match "^dflash([1-9]|1[0-5])$") {
    $SPEC_ARGS += @("--spec", "dflash", "--draft-tokens", $Matches[1])
}
elseif ($SPEC -ne "none" -and $SPEC -ne "mtp0") {
    Write-Error "invalid NINFER_SPEC: $SPEC (use none | mtp1..mtp5 | dflash1..dflash15)"
    exit 2
}
# dflash cannot be combined with vision (engine rejects it).
if ($SPEC -like "dflash*" -and $VISION -eq 1) {
    Write-Error "NINFER_SPEC=$SPEC cannot be combined with vision model $Model"
    exit 2
}

# Stability-test sampling profile (hardcoded process-level overrides). Applied to
# every request; a client may still override per-request via the API. NInfer has no
# repetition_penalty sampler, so the neutral 1.0 needs no flag.
$SAMPLING_ARGS = @(
    "--temperature", "0.7",
    "--top-p", "0.80",
    "--top-k", "20",
    "--min-p", "0.0",
    "--presence-penalty", "1.5"
)

$ARGS = @($ARTIFACT,
    "--host", $BIND_HOST,
    "--port", $PORT,
    "--model-id", $MODEL_ID,
    "--max-context", $MAX_CONTEXT,
    "--default-max-tokens", $MAX_GEN_TOKENS,
    "--kv-dtype", "int8",
    "--kv-capacity", "auto",
    "--max-concurrency", $Concurrency,
    "--request-log-jsonl", (Join-Path $LOG_DIR "ninfer-serve-$(Get-Date -Format 'yyyyMMdd').jsonl")
) + $SAMPLING_ARGS + $SPEC_ARGS
if ($VISION -eq 1) { $ARGS += "--vision" }

Write-Host "=================================================="
Write-Host "   NInfer server - $DESC"
Write-Host "   artifact:    $ARTIFACT"
Write-Host "   concurrency: $Concurrency"
Write-Host "   max context: $MAX_CONTEXT"
Write-Host "   spec:        $SPEC"
Write-Host "   sampling:    temp=0.7 top_p=0.80 top_k=20 min_p=0.0 presence=1.5"
Write-Host "   max gen:     $MAX_GEN_TOKENS tokens (default)"
Write-Host "   endpoint:    http://localhost:$PORT  (model id: $MODEL_ID)"
Write-Host "=================================================="
& $BIN @ARGS
exit $LASTEXITCODE
