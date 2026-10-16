# Components own targets and dependencies; source manifests remain explicit.
add_subdirectory(core)
add_subdirectory(artifact)
add_subdirectory(ops)

if(NINFER_BUILD_ENGINE)
  add_subdirectory(text)
  add_subdirectory(media)
  add_subdirectory(models)
  add_subdirectory(runtime)
endif()

if(NINFER_BUILD_PRODUCT_SUPPORT)
  add_subdirectory(product)
  add_subdirectory(serve)
endif()
