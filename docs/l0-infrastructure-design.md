# L0 Infrastructure Design — qwen3.6-ultraspeed

> Status: design (approved in brainstorm). Date: 2026-06-25.
> Scope: the **L0 infrastructure layer** — the model-agnostic, reusable foundation beneath the
> kernels (L1) and the model card (L2). See [`design.md`](design.md) for the project goals and
> [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) for the model specifics this
> infra is sized for.

---

## 1. Purpose & boundary

L0 provides device/memory/tensor plumbing for a **single-model, single-user, batch=1,
single-GPU, static-schedule** engine. It is **model-agnostic in code** but **parameterized by
`ModelConfig`** (the `constexpr` config lives in L2; L0 takes dims as parameters/template
args, so a future model reuses L0 unchanged).

**Design principles**
- **CUDA-only** (sm_120), single device, single compute stream, batch=1.
- **No external compute/runtime libraries** — CUDA Runtime + Driver API only. No
  cuBLAS / CUTLASS / CUB / Thrust. (All compute is hand-written in L1.)
- **Weights are device-resident** — uploaded once into VRAM (required for the decode GEMV
  bandwidth roofline; *not* host-mmapped at runtime).
- **No general allocator, no paging, no per-step `cudaMalloc`.** Almost all memory is
  allocated once at load; only the workspace is reused per step.
- **固化 (frozen):** the static schedule, the fixed weight-file layout, the cache shapes.
  **自由 (flexible):** dims are config-parameterized so L0 is reusable.

---

## 2. Prior-art study & the degrees-of-freedom verdict

We studied how vLLM (v1) and llama.cpp/ggml manage this, then kept only the freedoms our scope
needs. (Sources in §11.)

**How each manages it**
- **vLLM (v1):** `torch.Tensor` over PyTorch's caching allocator; **profiles** peak memory then
  gives the KV cache the remainder; KV is **paged** (`BlockPool` + per-request block tables +
  `slot_mapping`), and that same machinery is reused for GDN/Mamba state (≈1 page/seq in
  `"none"` mode); decode is a **captured CUDA graph** over **persistent I/O buffers**.
- **llama.cpp/ggml:** a universal `ggml_tensor` (`ne[]` sizes + `nb[]` byte strides;
  view/permute/reshape are metadata-only); a **`no_alloc` metadata context** separate from
  **backend payload buffers**; the workspace is `ggml_gallocr` — **measure the graph once →
  reserve one compute buffer → free-list reuse + in-place**; KV and recurrent (conv/ssm) state
  are **contiguous per-layer tensors** with ring-cell management; a multi-backend scheduler
  drives execution.

**Verdict — drop vs keep**

| Concern | Their generality | Ours |
|---|---|---|
| KV layout | paging: block tables + `slot_mapping` / ring `find_slot` | **contiguous, position-indexed**; append at `pos` |
| GDN state | paged "all/align" modes + `s_copy` gather | **one in-place conv+ssm buffer per layer** |
| Allocator | `BlockPool` / `ggml_gallocr` free-list + `dyn_tallocr` | **static regions + bump-reset workspace** |
| Sharing | prefix cache, eviction, watermark, preemption | **none** (single user) |
| Batch | variable buckets, micro-batch, `n_stream`, continuous batching | **batch=1, n_stream=1** |
| Backends | multi-backend sched, CPU fallback, op-probing, 40+ dtypes, ~96 ops, dynamic cgraph | **single CUDA path, fixed op set, hand-written schedule** |
| Loading | model-agnostic arch zoo, multi-split | **one fixed-file loader for our format** |
| Streams | async output, pipeline-parallel, TP/PP | **single compute stream** (+ optional load stream) |

**Ideas borrowed (kept):** ① thin **`ne/nb` tensor view** (zero-copy KV/state slices &
permutes); ② **descriptors-vs-payload** separation; ③ **measure-once → reserve-one-workspace +
in-place reuse**, *frozen* into the static schedule (workspace is a bump-reset arena, with a
static-overlap optimization left open); ④ **peak-activation sizing** + static budget; ⑤
**persistent I/O buffers** for the future decode CUDA graph; ⑥ **contiguous per-layer KV +
fixed conv/ssm state**; ⑦ one-shot **weight load → device VRAM**; ⑧ **kernel warmup before
graph capture**.

---

## 3. Memory model — `DeviceArena` (3 lifetime regions)

A small number of upfront `cudaMalloc` slabs, each carved by a bump pointer. No freeing of
individual allocations.

| Region | Lifetime | Contents | Reset? |
|---|---|---|---|
| **Weights** | persistent | all model weights (W4 packed + scales + sensitive bf16) | never |
| **Cache** | persistent | KV (16 full-attn layers) + GDN state (48 GDN layers), sized from `(max_context, dims)` | never |
| **Workspace** | transient | per-step activations / scratch | **`reset()` each step** (pointer→0) |

Plus a **pinned host staging buffer** for async H2D copies during weight load.

**Workspace strategy:** simple **reset bump-arena** — kernels bump-allocate scratch during a
step; the arena pointer resets to 0 at step end. Deterministic and CUDA-graph-safe (fixed base
+ deterministic bump order). At batch=1 the transient activations are tiny (decode) or a single
prefill peak, so the lack of lifetime-overlap reuse costs negligible memory. A precomputed
**static-offset overlay** (liveness-based) is a future optimization, only if long-prefill
activation peak ever pressures VRAM — not v1.

**Budget anchor** (@128K, from `design.md` §8): weights ~14–15 GB + KV ~8 GB + GDN state
~0.15 GB + workspace ~1–2 GB ≈ 24–26 GB of 32 GB.

```cpp
class DeviceArena {                 // one per region
public:
  explicit DeviceArena(size_t capacity_bytes);   // single cudaMalloc
  Tensor alloc(DType dt, std::initializer_list<int> shape, size_t align = 256);
  void   reset();                   // workspace only: offset_ = 0
  size_t used()  const; size_t capacity() const;
private:
  void* base_; size_t cap_; size_t off_;
};
```

---

## 4. Tensor, DType, QuantWeight

```cpp
enum class DType : uint8_t { BF16, FP32, I32, U8 };   // U8 = packed-4bit storage byte
size_t dtype_size(DType);                              // bytes per element (U8 = 1)

struct Tensor {                  // non-owning VIEW (no ownership, no alloc)
  void*   data  = nullptr;
  DType   dtype = DType::BF16;
  int32_t ne[4] = {1,1,1,1};     // sizes; ne[0] is the fastest-varying dim
  int64_t nb[4] = {0,0,0,0};     // byte strides; contiguous: nb[0]=dtype_size, nb[i]=nb[i-1]*ne[i-1]
  // metadata-only helpers (no copy): view/slice(dim,start,len), permute, reshape,
  // numel(), bytes(), is_contiguous().
};

struct QuantWeight {             // a W4 linear weight (logical [N,K])
  const void* qdata;             // packed 4-bit values (kernel-optimal layout)
  const void* scales;            // per-group scales
  int32_t n, k, group;           // logical out/in dims, group size
  // layout tag (enum) describing the packing, set by the offline packer
};
```

- Most dims are `constexpr` (from `ModelConfig`); `Tensor` carries them at runtime so kernels
  and KV/state slicing stay generic.
- `QuantWeight` is the descriptor the W4A16 GEMM consumes; its layout is produced offline
  (see `design.md` §10 weight pipeline) and is part of the fixed file.

---

## 5. Components (generic; instantiated with `ModelConfig` dims)

### 5.1 DeviceContext
Owns device selection + properties, the **compute stream** (+ optional **load stream** for
overlapping H2D during load), error-checking, logging, and profiling hooks.
```cpp
struct DeviceContext {
  int          device;
  cudaStream_t stream;        // single compute stream
  cudaStream_t load_stream;   // optional, for weight upload overlap
  // device props (sm, VRAM), NVTX range helpers, timers
};
#define CUDA_CHECK(expr) /* check cudaError_t, log file:line, abort */
```

### 5.2 WeightStore / Loader
Reads the **one fixed q5090 weight file**, validates its header and tensor table against the
`constexpr` config, and uploads selected modules into the **Weights** region. It does not recompute
payload CRCs during normal model load; CRC remains an offline converter/auditor field.
Exposes typed `Tensor` / `QuantWeight` views by name or by q5090 module/source id.
```cpp
class WeightStore {
public:
  WeightStore();
  void load(const char* path, DeviceArena& weights, DeviceContext& ctx,
            const LoadOptions& options = {});

  const Tensor*      tensor (string_view name) const;
  const QuantWeight* qweight(string_view name) const;
  const Tensor*      tensor (ModuleKind module, uint32_t source_kind, uint32_t source_layer) const;
  const QuantWeight* qweight(ModuleKind module, uint32_t source_kind, uint32_t source_layer) const;
};
```
- q5090 carries a module index (`TEXT_CORE`, `MTP_DRAFT`, `VISION_ENCODER`), tensor index,
  string table, payload spans, qtypes/layouts, and CRC32 per payload. Format spec:
  [`q5090_packed_file_format_v1.md`](q5090_packed_file_format_v1.md).
- `LoadOptions` defaults to TEXT only; MTP and vision payloads are opt-in. Unselected module
  metadata remains queryable, but payload pointers stay null.
- Upload path: read file → strict parse/CRC → async `cudaMemcpy` on `load_stream` → tensors live
  in Weights region. Optional progress callbacks report read, CRC, and upload phases for large
  q5090 files.

### 5.3 KVCache (full-attention layers)
Contiguous per-layer K/V in the Cache region; **position-indexed**, append-only.
```cpp
struct KVCache {                 // sized for 16 full-attn layers
  Tensor k[16];                  // per layer: [Hkv=4, dh=256, max_ctx]  (exact layout per attn kernel)
  Tensor v[16];
  uint32_t pos = 0;              // current sequence length
  // append(layer, k_new, v_new) writes at slot `pos`; advance pos once per step
};
```

### 5.4 StateStore (GDN linear-attention layers)
Fixed-size, context-independent conv + SSM state; **in-place** update.
```cpp
struct GdnState {                // sized for 48 GDN layers
  Tensor conv[48];               // per layer: [conv_dim=10240, W-1=3]
  Tensor ssm [48];               // per layer: [Hv=48, dv=128, dk=128], FP32
};
```

### 5.5 WorkspaceArena
The bump-reset arena (§3) over the Workspace region; kernels request typed scratch `Tensor`s;
temporary lifetimes use the arena-owned RAII guard returned by `scope()`, including nested scopes.
Raw cursor marks and rewinds are not public. The model card calls `reset()` at the end of each
prefill/decode step. Scope destruction only restores the host cursor; it does not synchronize CUDA,
so reuse remains ordered by the owning stream.

---

## 6. Lifecycle

```
init:    DeviceContext (device, streams, props)
         DeviceArena weights / cache / workspace  (cudaMalloc; sizes from config + max_context)
load:    WeightStore.load(file)  -> validate header vs config -> upload into Weights region
size:    KVCache + GdnState placed in Cache region (from max_context, dims)
run:     per prefill/decode step:
           model card issues kernels, requesting scratch from WorkspaceArena
           full-attn layers append to KVCache at `pos`; GDN layers update GdnState in place
           WorkspaceArena.reset()
shutdown: free the few region slabs
```

---

## 7. Explicitly NOT in L0 (dropped per §2)
Paging / block tables / `slot_mapping`; general allocators & free-lists; prefix caching;
eviction / preemption / watermark; multi-backend scheduler + CPU fallback; op-support probing;
variable-batch buckets / micro-batching; multi-stream async output / pipeline parallel / TP /
PP; model-agnostic arch loader. (These exist in vLLM/llama.cpp only to serve generality we
don't have.)

---

## 8. File layout
```
include/qus/core/
  device.h        # DeviceContext, CUDA_CHECK, NVTX/timing
  dtype.h         # DType + sizes
  tensor.h        # Tensor view, QuantWeight
  arena.h         # DeviceArena (regions) + WorkspaceArena
  weight_store.h  # WeightStore / loader
  kv_cache.h      # KVCache
  state_store.h   # GdnState
src/core/
  *.cpp / *.cu    # implementations
```
`ModelConfig` (dims) is a `constexpr` header in the **model module (L2)**, not L0; L0 takes
dims as parameters.

---

## 9. Open / deferred
- **Static-offset workspace overlay** (liveness-based) — only if prefill activation peak
  pressures VRAM.
- **Pinned-host double-buffering** for faster load — minor; revisit if load time matters.
- **fp8 KV / 256K context** — design.md deferred items; KVCache must accept a dtype parameter
  so this is a config change, not a redesign.
- **CUDA-graph I/O buffers** — persistent input/output `Tensor`s for the future decode graph;
  reserve their slots in the Workspace/Cache region when that work lands.
- **Exact KV physical layout** (`[Hkv,dh,ctx]` vs `[ctx,Hkv,dh]`) — pinned by the attention
  kernel's access pattern (L1 decision).

---

## 10. Borrowed-idea provenance (quick map)
| Our mechanism | Borrowed from |
|---|---|
| `Tensor` with `ne/nb` metadata views | ggml `ggml_tensor` |
| descriptors vs payload buffers | ggml `no_alloc` ctx + backend buffer |
| measure→reserve one workspace, in-place reuse | ggml `ggml_gallocr` (frozen to static) |
| peak-activation sizing + static budget | vLLM `determine_available_memory` |
| persistent I/O buffers for CUDA graphs | vLLM v1 model runner |
| contiguous per-layer KV + fixed recurrent state | both (llama.cpp KV/recurrent caches) |
| one-shot weight upload to VRAM | both |

---

## 11. Sources
- vLLM v1 infra study (memory profiling, BlockPool/paging, Mamba state, CUDA graphs,
  loader, streams) — `/home/neroued/vllm/vllm/v1/...`.
- llama.cpp/ggml infra study (`ggml_tensor`, `ggml_context` + backend buffers,
  `ggml-alloc`/`ggml_gallocr`, `llama_kv_cache`, `llama_memory_recurrent`,
  `ggml_backend_sched`, GGUF loader) — `/home/neroued/llama.cpp/...`.
