# Components own targets and dependencies; source manifests remain explicit.
add_subdirectory(core)
add_subdirectory(artifact)
add_subdirectory(text)
add_subdirectory(media)
add_subdirectory(ops)
add_subdirectory(models)
add_subdirectory(runtime)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  add_subdirectory(product)
  add_subdirectory(serve)
endif()
