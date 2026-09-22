# Quick repro test: start server, wait for health, send small (~1024-token) payload.
$ErrorActionPreference = "Stop"
# Resolve build tree, models, DLLs, logs, and vendored payloads portably.
. (Join-Path $PSScriptRoot "win_paths.ps1")
$root = $WORKSPACE_ROOT
if (Test-Path $FFMPEG_BIN) { $env:PATH = "$FFMPEG_BIN;$env:PATH" }
if (Test-Path $CURL_BIN)   { $env:PATH = "$CURL_BIN;$env:PATH" }
Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
$log = Join-Path $LOG_DIR "repro_run.log"
Remove-Item $log -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $NINFER_BIN `
  -ArgumentList @((Join-Path $MODELS "qwen3_8_27b_nvfp4.ninfer"),"--host","0.0.0.0","--port","8080","--model-id","256k","--max-context","131072","--default-max-tokens","65536","--kv-dtype","int8","--kv-capacity","auto","--max-concurrency","1","--spec","mtp","--draft-tokens","3","--lm-head-draft") `
  -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError (Join-Path $LOG_DIR "repro_err.log") -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Seconds 2
    try { $h = (Invoke-WebRequest -Uri http://localhost:8080/health -UseBasicParsing -TimeoutSec 5).Content; if ($h -match "ok") { $ok = $true; break } } catch { }
    if ($proc.HasExited) { Write-Output "SERVER DIED DURING STARTUP"; break }
}
Write-Output ("health_ok=" + $ok)
if ($ok) {
    $body = [System.IO.File]::ReadAllText((Join-Path $PAYLOAD_DIR "small.json"))
    try {
        $r = Invoke-WebRequest -Uri http://localhost:8080/v1/chat/completions -Method POST -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 300
        Write-Output ("request_ok len=" + $r.Content.Length)
        Write-Output ("content_head=[" + $r.Content.Substring(0, [Math]::Min(400, $r.Content.Length)) + "]")
    } catch { Write-Output ("request_err=" + $_.Exception.Message) }
}
Start-Sleep -Seconds 2
Write-Output "=== log tail ==="
Get-Content $log -Tail 15
Get-Process ninfer-serve -ErrorAction SilentlyContinue | Stop-Process -Force
if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
