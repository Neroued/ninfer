target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/measurement.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/decoder_state.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/state_image.cpp"

  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program_impl.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/context_work.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix_identity.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/round_buffers.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vision_control.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/graphs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefill.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/decode.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/grammar_masks.cpp"

  "${CMAKE_CURRENT_LIST_DIR}/program/planning/startup.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/planning/graph_profiles.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/planning/request_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/planning/source.cpp"

  "${CMAKE_CURRENT_LIST_DIR}/program/storage/sequence.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/storage/checkpoints.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/storage/draft_context.cpp"

  "${CMAKE_CURRENT_LIST_DIR}/program/transactions/context_transaction.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/transactions/binding.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/transactions/capture.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/transactions/commit.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/transactions/pause.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/transactions/reclaim.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/transactions/replay.cpp"

  "${CMAKE_CURRENT_LIST_DIR}/program/speculative/mtp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/speculative/target_verification.cpp"
)
