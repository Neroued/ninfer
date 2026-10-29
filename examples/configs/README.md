# Deployment configurations

These TOML files are starting points for the product CLI and HTTP server. All enable Vision,
FP8 KV, CUDA Graph decode, MTP3 and the optimized proposal head. The artifact must contain those
components. Model sampling defaults are retained. Download the corresponding artifacts from the
[weight table](../../README.md#get-weights); Flash Next and GLM require every storage part.

| Configuration | Hardware | Per-request context limit | Active requests |
|---|---|---:|---:|
| [Qwen3.8-27B NVFP4](rtx5090-qwen3_8-27b-nvfp4.toml) | RTX 5090 | 262,144 | 2 |
| [Qwen3.8-27B groupwise](rtx5090-qwen3_8-27b-groupwise.toml) | RTX 5090 | 262,144 | 2 |
| [Flash Next](spark-qwen3_8-flash-next.toml) | One Spark | 262,144 | 8 |
| [Flash Next TP2](spark2-qwen3_8-flash-next.toml) | Two Sparks | 262,144 | 8 |
| [GLM-5.3-Flash TP2/ADP2](spark2-glm5_3-flash.toml) | Two Sparks | 1,048,576 | 8 |

`context_capacity = "auto"` uses the available startup budget for the shared KV/State pool.
The context limit is a logical request ceiling, not a promise that all active requests can reach
it together. Read the startup pool size and token-capacity estimate for the actual deployment.
MTP3 is a useful starting configuration; the best draft count depends on workload and concurrency.

## One device

Copy an example to the repository root and edit `model.path`:

```bash
cp examples/configs/rtx5090-qwen3_8-27b-nvfp4.toml ninfer.local.toml
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml
```

Use `build/dgx-spark-release/apps/` on Spark. For a single CLI request, use the same configuration:

```bash
./build/rtx5090-release/apps/ninfer --config ninfer.local.toml \
  --prompt "Explain speculative decoding." --max-tokens 2048
```

The service listens on loopback by default. Set `server.host = "0.0.0.0"` for LAN access and
configure `server.api_key` when desired. Use the configured `server.model_id` in API requests.
CLI ignores server-only settings. Explicit command-line options override the file.

## Two Sparks

Build the same source revision on both nodes with NCCL and distributed execution enabled:

```bash
cmake --preset dgx-spark-release -DNINFER_ENABLE_DISTRIBUTED=ON
cmake --build build/dgx-spark-release -j
```

Copy one of the `spark2-*.toml` examples to a personal file. Replace:

- `model.path`: the artifact entry file visible to the coordinator;
- each node's absolute `worker` and `artifact` paths; place all artifact parts on both nodes;
- `control_host`: node 0's address reachable from node 1;
- node 1's `ssh` destination, using noninteractive SSH access;
- optional per-node NCCL environment entries with the actual network interface and RDMA HCA names.

The addresses `192.0.2.10` and `192.0.2.11` are placeholders. The worker's shared libraries must
remain beside its build's `apps/` directory in `lib/`. Nodes use the same artifact contents;
loading selects each rank's shard.

Run once on node 0:

```bash
./build/dgx-spark-release/apps/ninfer-serve --config ninfer.local.toml
```

The coordinator starts one local worker and the remote worker over SSH, assigns their ranks,
and shuts them down with the Engine. Clients use the coordinator's single HTTP endpoint.
Control traffic uses `control_host`; NCCL chooses the GPU communication transport from the node
configuration. Set the commented NCCL entries when the default selection is not your intended link.

Both dual-node examples use expert TP (`ep` defaults to 1). Set `ep = 2` under
`[distributed.parallel]` to use whole-expert EP while retaining the other TP partitions.
Keep `adp = 1` for Flash Next and `adp = 2` for GLM.

The [startup reference](../../docs/startup-configuration.md) describes every setting. The root
[ninfer.example.toml](../../ninfer.example.toml) is the complete field example. Personal TOML files
outside this directory remain ignored by Git. Relative file paths are resolved beside the config;
there is no shell or environment-variable expansion.
