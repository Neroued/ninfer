add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  reader.cpp
  backing.cpp
  binder.cpp
  placement.cpp
  encoded_slice.cpp
  materializer.cpp
  views.cpp
)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json)
