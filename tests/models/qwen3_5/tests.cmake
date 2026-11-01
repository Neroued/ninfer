ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

# Qwen4Exp (Qwen3.8-Flash-Next): synthetic ModelOpt checkpoint -> both Flash-Next recipes -> load.
# The executable alone skips (77); the interop driver supplies artifacts and expectations.
ninfer_add_test(ninfer_qwen4_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_qwen4.cpp"
  LIBRARIES ninfer_model_loading)

# Qwen4Exp at tp 2 (T30): placement of every parameter of a real-geometry artifact, per-rank
# bytes, and (two devices) every device parent byte-compared with the host-applied slice. Skips
# (77) without --artifact.
ninfer_add_test(ninfer_qwen4_tp2_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_qwen4_tp2.cpp"
  LIBRARIES ninfer_model_loading)

add_test(NAME ninfer_qwen4_loading_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/qwen4_load_interop.py"
    $<TARGET_FILE:ninfer_qwen4_loading_test>)

set_tests_properties(
  ninfer_qwen3_5_loading_real_test
  ninfer_qwen4_loading_test
  ninfer_qwen4_tp2_loading_test
  ninfer_qwen4_loading_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Qwen4Exp routed-expert backends: the offload backend's windows, fills, misses, executor failure
# and MTP-layer logs over synthetic banks (GPU, AVX-512 VNNI; skips 77 otherwise), and the window
# protocol's error paths (CPU only).
ninfer_add_test(ninfer_qwen4_offload_experts_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qwen4_offload_experts.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_test(ninfer_qwen4_draft_head_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qwen4_draft_head.cpp"
  LIBRARIES ninfer_model_runtime)

ninfer_add_test(ninfer_qwen4_expert_window_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qwen4_expert_window.cpp"
  LIBRARIES ninfer_model_runtime)

set_tests_properties(
  ninfer_qwen4_offload_experts_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_context_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_engine ninfer_model_loading)

set_tests_properties(
  ninfer_qwen3_5_prefix_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Qwen4Exp at tp 2 against tp 1 through the Engine (T31). Skips (77) without
# NINFER_TEST_QWEN4_ARTIFACT (the synthetic fn4) or without two devices.
ninfer_add_test(ninfer_qwen4_tp2_engine_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_qwen4_tp2.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen4_tp2_engine_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Qwen4Exp MTP rounds through the Engine (T37): MTP == no-draft == oracle drafts with mapped and
# offloaded experts, and a cancelled row. Skips (77) without NINFER_TEST_QWEN4_ARTIFACT (a
# synthetic artifact with an MTP layer: tools/capy_flash/make_test_artifact.sh).
ninfer_add_test(ninfer_qwen4_mtp_engine_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_qwen4_mtp.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen4_mtp_engine_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Qwen4Exp with Vision at tp 2 and tp 1 (MTP, offload experts): text invariance, MTP == no-draft
# on an image prompt and its resumed second turn, tower copy parity across ranks. Skips (77)
# without NINFER_TEST_QWEN4_VISION_ARTIFACT (a synthetic artifact with Vision and MTP components).
ninfer_add_test(ninfer_qwen4_vision_engine_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_qwen4_vision.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen4_vision_engine_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine ninfer_model_loading ninfer_core)

add_test(NAME ninfer_qwen3_5_score_tp2_real_test
  COMMAND ninfer_qwen3_5_score_real_test tp2)

set_tests_properties(
  ninfer_qwen3_5_score_real_test
  ninfer_qwen3_5_score_tp2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_runtime ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_vision_workspace_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_engine ninfer_model_loading)

set_tests_properties(
  ninfer_qwen3_5_dflash2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_dflash_prefill_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_shard_map_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_shard_map.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen4_shard_map_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_shard_map_qwen4.cpp"
  LIBRARIES ninfer_model_loading ninfer_model_runtime)

ninfer_add_test(ninfer_qwen3_5_sharded_load_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sharded_load_real.cpp"
  LIBRARIES ninfer_model_runtime)

add_test(NAME ninfer_qwen3_5_sharded_load_mtp_real_test
  COMMAND ninfer_qwen3_5_sharded_load_real_test mtp)

set_tests_properties(
  ninfer_qwen3_5_sharded_load_real_test
  ninfer_qwen3_5_sharded_load_mtp_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_text_context_tp2_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_text_context_tp2.cpp"
  LIBRARIES ninfer_model_runtime)

add_test(NAME ninfer_qwen3_5_text_context_tp2_real_test
  COMMAND ninfer_qwen3_5_text_context_tp2_test real)

set_tests_properties(
  ninfer_qwen3_5_text_context_tp2_test
  ninfer_qwen3_5_text_context_tp2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_engine_tp2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tp2_real.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_engine_tp2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_engine_mtp_tp2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_mtp_tp2_real.cpp"
  LIBRARIES ninfer_engine ninfer_core)

add_test(NAME ninfer_qwen3_5_engine_mtp_tp2_optimized_real_test
  COMMAND ninfer_qwen3_5_engine_mtp_tp2_real_test optimized)

set_tests_properties(
  ninfer_qwen3_5_engine_mtp_tp2_real_test
  ninfer_qwen3_5_engine_mtp_tp2_optimized_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_engine_dflash2_tp2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_tp2_real.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_engine_dflash2_tp2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_engine_vision_tp2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_vision_tp2_real.cpp"
  LIBRARIES ninfer_engine ninfer_core)

add_test(NAME ninfer_qwen3_5_engine_vision_tp2_mtp_real_test
  COMMAND ninfer_qwen3_5_engine_vision_tp2_real_test mtp)

add_test(NAME ninfer_qwen3_5_engine_vision_tp2_dflash2_real_test
  COMMAND ninfer_qwen3_5_engine_vision_tp2_real_test dflash2)

set_tests_properties(
  ninfer_qwen3_5_engine_vision_tp2_real_test
  ninfer_qwen3_5_engine_vision_tp2_mtp_real_test
  ninfer_qwen3_5_engine_vision_tp2_dflash2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_ple_gather_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_gather.cpp"
  LIBRARIES ninfer_model_runtime)

# Qwen4Exp validation driver (not a CTest): Engine scoring / greedy generation over fixed ids for
# a teacher-forcing harness (tfdump/1 format, src/models/qwen3_5/execution/qwen4_tfdump.h).
add_executable(ninfer_qwen4_tf_driver "${CMAKE_CURRENT_LIST_DIR}/qwen4_tf_driver.cpp")
target_link_libraries(ninfer_qwen4_tf_driver PRIVATE ninfer_engine)

# capy-flash v0.7: the GGUF slot arena of the offload backend against the expert cache's slot
# numbering (CPU only).
ninfer_add_test(ninfer_qwen4_gguf_pool_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qwen4_gguf_pool.cpp"
  LIBRARIES ninfer_model_runtime)
