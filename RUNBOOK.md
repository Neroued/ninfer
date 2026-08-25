# Runbook: Warmup Fail-Fast and Exception Logging Validation (Issue #4)

## Overview

This runbook documents the operational verification procedures for:
1. **Crash / Terminate Logging**: Ensuring unhandled exceptions escaping thread or server boundaries print 	ypeid(error).name() and rror.what() directly to stderr before aborting (preventing silent hlt / general protection fault under Docker PID 1).
2. **Warmup Fail-Fast**: Ensuring any exception during startup warmup (e.g. OOM, corrupted prefix cache, invalid batch allocation) fails the process immediately with non-zero exit code (1) instead of continuing in a zombie state returning 503 errors.
3. **Warmup Timeout Override**: Ensuring startup warmup uses an explicit 60-second budget rather than short client-facing --pending-timeout-ms.
4. **Auto KV-Capacity Bounding**: Clarifying --kv-capacity auto description in --help to state (bounded by max-context * max-concurrency).

---

## 1. Automated Unit Tests

All serve unit tests run and pass inside the build container:

`ash
docker run --rm -v "P:\NInfer.gemini:/src" -w /src ninfer:test-build bash -c "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs ctest --test-dir build-linux -R 'ninfer_(serve_options|http_error_handler|openai_schema|responses_schema|response_store|anthropic_schema|tool_call_parser|request_log|kv_capacity)_test' --output-on-failure"
`

### Verified Test Cases:
- 
infer_serve_options_test: Verifies --help text contains (bounded by max-context * max-concurrency) for --kv-capacity auto.
- 
infer_http_error_handler_test: Verifies HTTP error JSON mapping.
- 
infer_kv_capacity_test: Verifies sequence capacity curve and page allocation bounds.
- 
infer_openai_schema_test, 
infer_anthropic_schema_test, 
infer_responses_schema_test, 
infer_response_store_test, 
infer_tool_call_parser_test, 
infer_request_log_test: 100% passing.

---

## 2. Induced Failure & Error Path Procedures (GPU Host Verification)

When scheduled in a maintenance window with GPU allocation lock (C:\Users\igorl\.ninfer-gpu.lock):

### Procedure A: Induce Warmup Failure (OOM / Allocation Fault)
Run 
infer-serve with --prefix-cache-mib set higher than available GPU memory, e.g.:
`ash
./build-linux/apps/ninfer-serve --model-path out/qwen3_8_27b.ninfer --kv-capacity auto --prefix-cache-mib 60000 --port 8088
`
**Expected Outcome**:
- Startup logs: atal: warmup generation failed: ... (or atal: failed to allocate prefix cache ...).
- Process terminates immediately with exit code 1.
- Server port 8088 is never bound/left in zombie state.

### Procedure B: Verify Clean Warmup & Normal Boot
Run 
infer-serve with normal options:
`ash
./build-linux/apps/ninfer-serve --model-path out/qwen3_8_27b.ninfer --kv-capacity auto --port 8088
`
**Expected Outcome**:
- Console logs:
  `	ext
  info: warming up generation service
  info: generation service ready
  info: listening on 0.0.0.0:8088
  `
- curl http://127.0.0.1:8088/health or /v1/models returns 200 OK.

### Procedure C: Verify PID 1 Terminate Logging Handler
In a test container running without a custom init system:
- Trigger an unhandled exception in a worker thread.
- **Expected Outcome**:
  - atal: unhandled exception: <type_name>: <message> is emitted to stderr.
  - Process exits cleanly via std::abort().
