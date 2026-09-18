target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/text.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/mtp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/dflash.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/dflash2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/resources.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/prepare.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/resources.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tokenizer.cpp"
)
