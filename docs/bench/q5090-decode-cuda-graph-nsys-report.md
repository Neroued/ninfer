# Q5090 Decode CUDA Graph Nsys Report

> Date: 2026-07-01  
> Code commit: `c0ba1a8` (`feat(q5090): capture/replay decode as one CUDA graph`)  
> Scope: Qwen3.6-27B q5090 mixed v3 long decode on one RTX 5090 after CUDA graph
> capture/replay for the static decode step.

## Workload

```bash
nsys profile --force-overwrite=true --stats=false --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  -o profiles/nsys-decode-cuda-graph/heat_fem_graph \
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
    --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
    --prompt "怎么用fem方法求解heat equation。" \
    --max-context 8192 --max-new 2916
```

Program timing from stderr:

| item | value |
| --- | ---: |
| engine load total | `12.555 s` |
| prefill | `1.771 s` |
| decode | `52.445 s` |
| generated tokens | `2868` |
| model elapsed | `54.215 s` |
| throughput | `52.90 tok/s` |

The run stopped before `--max-new 2916`; generated-token count includes the prefill-produced token,
so the decode loop executed `2867` steps. The first decode step is the eager warmup. The remaining
`2866` decode steps use CUDA graph launch, with the second step also doing graph capture before the
first graph launch.

## Artifacts

| artifact | path |
| --- | --- |
| Nsight Systems report | `profiles/nsys-decode-cuda-graph/heat_fem_graph.nsys-rep` |
| SQLite export | `profiles/nsys-decode-cuda-graph/heat_fem_graph.sqlite` |
| Generated summary | `profiles/nsys-decode-cuda-graph/heat_fem_graph.summary.md` |
| Decode metrics | `profiles/nsys-decode-cuda-graph/heat_fem_graph.decode_metrics.csv` |
| CUDA API summary | `profiles/nsys-decode-cuda-graph/heat_fem_graph.cuda_api_sum_cuda_api_sum.csv` |
| CUDA kernel summary | `profiles/nsys-decode-cuda-graph/heat_fem_graph.cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv` |
| CUDA memop summary | `profiles/nsys-decode-cuda-graph/heat_fem_graph.cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv` |

The `.nsys-rep` and SQLite files are local profiling artifacts. The committed evidence is this
report plus the lightweight exported summary/CSV files.

Export commands:

```bash
python3 ~/.codex/skills/nsys-inference-analysis/scripts/nsys_inference_summary.py \
  profiles/nsys-decode-cuda-graph/heat_fem_graph.nsys-rep \
  --out profiles/nsys-decode-cuda-graph/heat_fem_graph.summary.md

nsys stats --force-overwrite=true --report cuda_api_sum --format csv \
  --output profiles/nsys-decode-cuda-graph/heat_fem_graph.cuda_api_sum \
  profiles/nsys-decode-cuda-graph/heat_fem_graph.nsys-rep

nsys stats --force-overwrite=true --report cuda_gpu_kern_sum --format csv \
  --output profiles/nsys-decode-cuda-graph/heat_fem_graph.cuda_gpu_kern_sum \
  profiles/nsys-decode-cuda-graph/heat_fem_graph.nsys-rep

nsys stats --force-overwrite=true --report cuda_gpu_mem_time_sum --format csv \
  --output profiles/nsys-decode-cuda-graph/heat_fem_graph.cuda_gpu_mem_time_sum \
  profiles/nsys-decode-cuda-graph/heat_fem_graph.nsys-rep
```

## Launch Collapse

Decode-window metrics from `heat_fem_graph.decode_metrics.csv`:

| metric | value |
| --- | ---: |
| decode NVTX wall | `52.444800 s` |
| decode steps | `2867` |
| CUDA graph trace events | `2866` |
| CUDA graph trace total | `51.357582 s` |
| CUDA graph trace share of decode | `97.927%` |
| `cudaGraphLaunch` calls | `2866` |
| `cudaLaunchKernel` calls inside decode | `2666` |
| visible child kernel events | `1333` |

Nsight Systems records graph replay as `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`; it no longer materializes
every replayed child kernel as a `cudaLaunchKernel` API call. The remaining decode-window
`cudaLaunchKernel` calls are bounded setup work from the first eager warmup decode and the graph
capture body, not per-token steady replay. The graph trace total is replay-duration evidence for
the captured decode graph executions; it is not child-kernel summed time.

The immediate pre-graph tuned baseline is the `previous_tuned` run in
`profiles/nsys-long-decode/heat_fem_max2916_current_vs_tuned.decode_analysis.txt`. That same file
also contains an older `current` run before the final tuned baseline. Use `previous_tuned` for the
graph-specific wall comparison:

| metric | historical `current` | immediate `previous_tuned` | CUDA graph |
| --- | ---: | ---: | ---: |
| decode steps | `2915` | `2915` | `2867` |
| decode wall per step | `19.694 ms` | `18.782 ms` | `18.293 ms` |
| decode steps/s | `50.776` | `53.242` | `54.667` |
| host kernel launches per step | `1429` | `1333` | `~1 cudaGraphLaunch` steady replay |
| decode `cudaLaunchKernel` API total | `20.134 s` | `18.739 s` | `0.0106 s` bounded warmup/capture |

Normalized decode wall improves by `2.61%` per step versus the immediate pre-graph tuned baseline.
Against the older historical `current` row, the normalized delta is `7.12%`; that is useful context
but not the graph-only gain. The same-output 256-token parity gate also showed decode wall
improving from `5.300 s` eager to `4.551 s` with graph enabled.

## Verification

```bash
diff /tmp/graph.out /tmp/eager.out
compute-sanitizer --tool memcheck ./build/src/qus \
  out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
  --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --prompt "怎么用fem方法求解heat equation。" \
  --max-context 8192 --max-new 32
ctest --test-dir build
```

Results:

| gate | result |
| --- | --- |
| graph vs eager 256-token stdout diff | no diff; both files `794 B` |
| compute-sanitizer replay memcheck | `ERROR SUMMARY: 0 errors` |
| full regression suite | `33/33` tests passed |
