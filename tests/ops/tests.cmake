set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  rmsnorm_pack_tail
  gated_rmsnorm
  l2norm
  gated_delta_net
  kimi_delta_attention
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  sampling
  scalar
  cast
  prepare_ragged_prefix
  scatter
  scatter_bf16_batch
  target_logprobs
  position
  allreduce
  allreduce_f32
  linear_split
  output_head_split
  proposal_head_split
  proposal_topk_split
  dflash2_head_blocks
  attention_headlocal
  attn_input_proj_split
  gdn_projections_split
  gdn_headsplit
  linear_swiglu_split
  linear_add_split
  hyper_connection
  hyper_connection_split
  ple
  moe_slots
  moe_grouped
  moe_route
  moe_predict)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES ninfer_ops)
endforeach()

# The SIMT router logits (NINFER_QWEN4_ROUTER_MMA=0; tensor cores are the default, T183) through the same test.
add_test(NAME ninfer_moe_route_simt_test COMMAND ninfer_moe_route_test)
set_tests_properties(ninfer_moe_route_simt_test PROPERTIES ENVIRONMENT "NINFER_QWEN4_ROUTER_MMA=0")

# The T62 exchange receives by kernel on a pair without peer access by default (since v0.11.1); the same test
# with NINFER_TP_XIN_KERNEL=0 keeps the copy-engine path covered. Skips with fewer than two GPUs like the test.
add_test(NAME ninfer_allreduce_f32_xin_test COMMAND ninfer_allreduce_f32_test)
set_tests_properties(ninfer_allreduce_f32_xin_test PROPERTIES ENVIRONMENT "NINFER_TP_XIN_KERNEL=1"
                     SKIP_RETURN_CODE 77)
add_test(NAME ninfer_allreduce_f32_copy_test COMMAND ninfer_allreduce_f32_test)
set_tests_properties(ninfer_allreduce_f32_copy_test PROPERTIES ENVIRONMENT "NINFER_TP_XIN_KERNEL=0"
                     SKIP_RETURN_CODE 77)
# Q8 prefill on int8 tensor cores (ops/linear/q8/q8_a8.h) against an exact float oracle.
ninfer_add_op_test(ninfer_q8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a8.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_topk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_topk.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_host_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_host_moe.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_host_nvfp4_isa_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_host_nvfp4_isa.cpp"
  LIBRARIES ninfer_ops)
# The whole host MoE test on the AVX2 kernels (skips where the CPU lacks AVX2).
add_test(NAME ninfer_host_moe_avx2_test COMMAND ninfer_host_moe_test)
set_tests_properties(ninfer_host_moe_avx2_test PROPERTIES ENVIRONMENT "NINFER_HOST_ISA=avx2")
# No AVX/AVX2/AVX-512 instruction outside the cpuid-dispatched host kernels (the binary must start on
# any x86-64 CPU; tools/check_host_isa.py). Scans the objects of the library build.
add_test(NAME ninfer_host_isa_guard_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/check_host_isa.py ${CMAKE_BINARY_DIR}/src)

ninfer_add_op_test(ninfer_host_moe_mailbox_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_host_moe_mailbox.cu"
  LIBRARIES ninfer_ops)
# The slice table check bounds the host gap between a layer's two slices (300 us; about 20 us alone):
# another process on the GPU or the CPU cores stretches it past a millisecond, so it runs alone.
set_tests_properties(ninfer_host_moe_mailbox_test PROPERTIES RUN_SERIAL TRUE)

# GGUF routed-expert Ops; real banks with NINFER_GGUF_BANKS, --bench for timings. The FP64 oracle
# is host code of a .cu file: keep its host compiler from contracting.
ninfer_add_op_test(ninfer_moe_slots_gguf_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_moe_slots_gguf.cu"
  LIBRARIES ninfer_ops)
target_compile_options(ninfer_moe_slots_gguf_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)

ninfer_add_op_test(ninfer_candidate_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_flash_next_tp2_mixers_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_flash_next_tp2_mixers.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_flash_next_tp2_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_flash_next_tp2_moe.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_flash_next_gdn_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_flash_next_gdn.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_sparse_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_attention.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_qsa_staging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qsa_staging.cpp"
  LIBRARIES ninfer_ops ninfer_core)

ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sliding_window_attention.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_append.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_rmsnorm_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_rmsnorm_rope_text_shard_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope_text_shard.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_context_kv_materialize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp"
  LIBRARIES ninfer_ops)

add_test(NAME ninfer_kv_cache_append_nvfp4_test
  COMMAND ninfer_kv_cache_append_test --nvfp4-only)

add_test(NAME ninfer_kv_cache_append_k8v4_test
  COMMAND ninfer_kv_cache_append_test --k8v4-only)

set_tests_properties(
  ninfer_kv_cache_append_nvfp4_test
  ninfer_kv_cache_append_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare_masked_block.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_dynamic_grouped_conv_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_dynamic_grouped_conv_add_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_dynamic_grouped_conv_add.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_snapshot.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_replay_fold.cpp"
  LIBRARIES ninfer_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/tests.cmake")

# T98 validation probe (docs/capy-flash/development-log.md): engine CPU W4A8 vs GPU W4A4 on real experts
add_executable(t98_expert_probe "${CMAKE_CURRENT_LIST_DIR}/t98_expert_probe.cu")
ninfer_test_includes(t98_expert_probe)
target_link_libraries(t98_expert_probe PRIVATE ninfer_ops)

# T103 check (docs/capy-flash/development-log.md): generic BF16 linear on Flash-Next shapes
add_executable(t103_bf16_generic "${CMAKE_CURRENT_LIST_DIR}/t103_bf16_generic.cu")
ninfer_test_includes(t103_bf16_generic)
target_link_libraries(t103_bf16_generic PRIVATE ninfer_ops)

# GGUF host kernels (fngine port) against ggml reference code; real GSQ-RCO rows when
# NINFER_GGUF_BANKS names a tools/convert/gguf_experts.py extract directory.
ninfer_add_test(ninfer_host_gguf_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_host_gguf.cpp"
  LIBRARIES ninfer_host_moe ninfer::json)
set_tests_properties(ninfer_host_gguf_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_op_oracle_options(ninfer_host_gguf_test)

# The host MoE executor on GGUF layers against an FP64 oracle (CPU only).
ninfer_add_test(ninfer_host_moe_gguf_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_host_moe_gguf.cpp"
  LIBRARIES ninfer_host_moe ninfer::json)
set_tests_properties(ninfer_host_moe_gguf_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_op_oracle_options(ninfer_host_moe_gguf_test)

# The same two GGUF tests on the AVX2 kernels (skip where the CPU lacks AVX2).
add_test(NAME ninfer_host_gguf_avx2_test COMMAND ninfer_host_gguf_test)
set_tests_properties(ninfer_host_gguf_avx2_test PROPERTIES ENVIRONMENT "NINFER_HOST_ISA=avx2" SKIP_RETURN_CODE 77)
add_test(NAME ninfer_host_moe_gguf_avx2_test COMMAND ninfer_host_moe_gguf_test)
set_tests_properties(ninfer_host_moe_gguf_avx2_test PROPERTIES ENVIRONMENT "NINFER_HOST_ISA=avx2" SKIP_RETURN_CODE 77)

# AVX2 GGUF kernels against the AVX-512 ones, bit for bit (needs a CPU with both; real rows with
# NINFER_GGUF_BANKS).
ninfer_add_test(ninfer_host_gguf_isa_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_host_gguf_isa.cpp"
  LIBRARIES ninfer_host_moe ninfer::json)
set_tests_properties(ninfer_host_gguf_isa_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_op_oracle_options(ninfer_host_gguf_isa_test)

# gate on the YaRN frequency table (ported 2026-10-24): factor 1 must reproduce the shipped
# kTextRopeInvFrequency exactly; a wrong table does not crash, it silently stops finding things.
ninfer_add_op_test(ninfer_yarn_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_yarn.cpp"
  LIBRARIES ninfer_ops)

# Metadata qualification uses the production collective without loading expert weights.
ninfer_add_op_test(ninfer_moe_grouped_scan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_moe_grouped_scan.cu"
  LIBRARIES ninfer_core)
