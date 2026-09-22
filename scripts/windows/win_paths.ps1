# win_paths.ps1 - shared path resolver for the Windows serve/test scripts.
#
# Dot-source this from a script that has already set $PSScriptRoot:
#     . (Join-Path $PSScriptRoot "win_paths.ps1")
#
# These scripts live at <repo>/scripts/windows/. REPO_ROOT is therefore two
# levels up. WORKSPACE_ROOT is the directory that holds the ninfer/ checkout
# (the parent of REPO_ROOT) - in this workspace that is inference-dev/, where the
# models/, third_party/, and debug/ siblings live.
# Resolution order for every path:  env override  ->  repo layout  ->  workspace-root fallback.
#
# Sets: $REPO_ROOT, $WORKSPACE_ROOT, $NINFER_BIN, $NINFER_COMPARE_BIN,
#       $MODELS, $LOG_DIR, $FFMPEG_BIN, $CURL_BIN, $PAYLOAD_DIR.

$script:REPO_ROOT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent   # .../scripts/windows -> repo
$script:WORKSPACE_ROOT = Split-Path $script:REPO_ROOT -Parent                # repo -> workspace root

function script:Resolve-NinferPath {
    param([string]$Override, [string]$Primary, [string]$Fallback)
    if ($Override) { return $Override }
    if (Test-Path $Primary) { return $Primary }
    return $Fallback
}

# Build tree. Resolution order:
#   1. $env:NINFER_BIN (explicit override)
#   2. the most-recently-built <repo>/build*/apps/ninfer-serve.exe (the CURRENT tree;
#      a frozen backup build\ is older, so the newest wins - never silently stale)
#   3. legacy workspace-root layout (<workspace>/ninfer/build-*/apps/...)
function script:Resolve-NinferBin {
    param([string]$Root, [string]$WorkspaceRoot)
    if ($env:NINFER_BIN -and (Test-Path $env:NINFER_BIN)) { return $env:NINFER_BIN }
    $cand = Get-ChildItem -Path (Join-Path $Root "build*") -Filter "ninfer-serve.exe" -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($cand) { return $cand.FullName }
    $cand = Get-ChildItem -Path (Join-Path $WorkspaceRoot "ninfer\build*") -Filter "ninfer-serve.exe" -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($cand) { return $cand.FullName }
    return (Join-Path $Root "build\apps\ninfer-serve.exe")   # last-resort path (may not exist)
}
$script:NINFER_BIN = Resolve-NinferBin -Root $script:REPO_ROOT -WorkspaceRoot $script:WORKSPACE_ROOT

# A/B candidate build (swap/watch -UseNew side); optional.
$script:NINFER_COMPARE_BIN = Resolve-NinferPath -Override $env:NINFER_COMPARE_BIN `
    -Primary (Join-Path $script:WORKSPACE_ROOT "ninfer-compare\build\apps\ninfer-serve.exe") `
    -Fallback (Join-Path $script:WORKSPACE_ROOT "ninfer-compare\build\apps\ninfer-serve.exe")

# Model artifacts: env override, else <repo>/models, else <workspace>/models.
$script:MODELS = Resolve-NinferPath -Override $env:NINFER_MODELS `
    -Primary (Join-Path $script:REPO_ROOT "models") `
    -Fallback (Join-Path $script:WORKSPACE_ROOT "models")

# Request/monitor logs: env override, else <repo>/debug, else <workspace>/debug.
$script:LOG_DIR = Resolve-NinferPath -Override $env:NINFER_LOG_DIR `
    -Primary (Join-Path $script:REPO_ROOT "debug") `
    -Fallback (Join-Path $script:WORKSPACE_ROOT "debug")

# Vendored test payloads (ship with the repo).
$script:PAYLOAD_DIR = Join-Path $PSScriptRoot "payloads"

# Third-party DLL dirs (ffmpeg + libcurl). NOT vendored in the repo (gitignored);
# supply your own and point NINFER_FFMPEG_BIN / NINFER_CURL_BIN at them, or drop
# them under <workspace>/third_party for the legacy layout.
$script:FFMPEG_BIN = Resolve-NinferPath -Override $env:NINFER_FFMPEG_BIN `
    -Primary (Join-Path $script:REPO_ROOT "third_party\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin") `
    -Fallback (Join-Path $script:WORKSPACE_ROOT "third_party\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin")
$script:CURL_BIN = Resolve-NinferPath -Override $env:NINFER_CURL_BIN `
    -Primary (Join-Path $script:REPO_ROOT "third_party\curl-inst\bin") `
    -Fallback (Join-Path $script:WORKSPACE_ROOT "third_party\curl-inst\bin")
