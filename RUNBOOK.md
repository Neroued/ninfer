# Runbook: Warmup Fail-Fast, Exception Logging, and Boot Watchdog (Issue #4)

## Overview

This runbook documents the operational verification procedures for:
1. **Crash / Terminate Logging**: Ensuring unhandled exceptions escaping thread or server boundaries print `typeid(error).name()` and `error.what()` directly through the console logger before calling `std::abort()` (preventing uninformative silent aborts under container PID 1).
2. **Warmup Fail-Fast**: Ensuring any exception during startup warmup (e.g., CUDA OOM, corrupted prefix cache, invalid batch allocation) terminates the process immediately with non-zero exit code (1) instead of leaving the process listening in an alive-but-503 zombie state.
3. **Pre-Listen Boot Watchdog**: A detached timer thread armed before warmup that terminates the process via `std::_Exit(1)` if boot does not reach the listening state within a configurable budget (default 120 s via `--boot-watchdog-timeout-s`), protecting against uncooperative GPU/driver wedges.
4. **Warmup Timeout Decoupling**: Ensuring startup warmup uses an explicit 60-second budget rather than the short client-facing `--pending-timeout-ms`.
5. **Auto KV-Capacity Bounding**: Clarifying `--kv-capacity auto` description in `--help` to state `(bounded by max-context * max-concurrency)`.

---

## 1. Automated Unit Tests (CPU Container Lane)

All serve unit tests run and pass inside the build container without requiring GPU hardware:

```bash
docker run --rm -v "P:\NInfer.gemini:/workspace" -w /workspace \
  -e LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs \
  ninfer:test-build bash -c \
  "cd /workspace/build && ctest --output-on-failure -R 'ninfer_(serve_options|http_error_handler|openai_schema|responses_schema|response_store|anthropic_schema|tool_call_parser|request_log|kv_capacity)_test'"
```

### Verified Test Cases:
- `ninfer_serve_options_test`: Verifies `--help` text contains `(bounded by max-context * max-concurrency)` for `--kv-capacity auto` and `--boot-watchdog-timeout-s` options.
- `ninfer_http_error_handler_test`: Verifies HTTP error JSON mapping.
- `ninfer_kv_capacity_test`: Verifies sequence capacity curve and page allocation bounds.
- `ninfer_openai_schema_test`, `ninfer_responses_schema_test`, `ninfer_response_store_test`, `ninfer_anthropic_schema_test`, `ninfer_tool_call_parser_test`, `ninfer_request_log_test`: 100% passing.

---

## 2. Induced Failure & Error Path Procedures (GPU Maintenance Window)

When executed in a coordinator-scheduled GPU maintenance window under the cross-agent lock protocol (`C:\Users\igorl\.ninfer-gpu.lock`):

### Procedure A: Induce Warmup Failure (OOM / Allocation Fault)
Run `ninfer-serve` with `--prefix-cache-mib` set higher than available GPU VRAM:
```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer --kv-capacity auto --prefix-cache-mib 60000 --port 8018
```
**Expected Observable Reality**:
- `httplib` binds the port and sets up the socket backlog at startup.
- Engine initialization or warmup throws `std::runtime_error("warmup generation failed: ...")` or allocation exception.
- Stderr log output:
  ```
  [YYYY-MM-DD HH:MM:SS.mmm] [error] ninfer-serve: warmup generation failed: ...
  ```
- The process does NOT enter `server.listen()` (the HTTP accept loop) and terminates immediately with exit code 1.
- Socket is closed upon process exit; no zombie 503 HTTP server remains running.

### Procedure B: Verify Clean Warmup & Normal Boot
Run `ninfer-serve` with standard production options:
```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer --kv-capacity auto --max-context 131072 --port 8018
```
**Expected Observable Reality**:
- Console logs:
  ```
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: loading model...
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: model loaded in ... s
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: KV capacity auto resolved=...
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: warming up...
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: listening on http://127.0.0.1:8018 (model id: ..., auth: disabled)
  ```
- Boot watchdog is cleanly disarmed upon reaching the listening state.
- `curl http://127.0.0.1:8018/v1/models` returns HTTP 200 OK with model descriptor.

### Procedure C: Verify PID 1 Terminate Logging Handler
In a test container running without a custom init system:
- Trigger an unhandled exception escaping a thread boundary.
**Expected Observable Reality**:
- Stderr log output:
  ```
  [YYYY-MM-DD HH:MM:SS.mmm] [error] ninfer-serve: terminate called after throwing <type_name>: <message>
  ```
- `std::abort()` terminates the process via `SIGABRT` (exit code 134, or container protection fault).

### Procedure D: Verify Boot Watchdog Hang Protection
To test the uncooperative hang fallback, run with a short watchdog timeout:
```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer --boot-watchdog-timeout-s 1 --port 8018
```
**Expected Observable Reality**:
- If model loading or warmup exceeds 1 second:
  ```
  [YYYY-MM-DD HH:MM:SS.mmm] [error] ninfer-serve: boot watchdog timeout (1 s) exceeded before reaching listening state; terminating process
  ```
- The watchdog terminates the process immediately via `std::_Exit(1)`.
