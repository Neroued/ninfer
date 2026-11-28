add_library(ninfer_deployment STATIC deployment.cpp topology.cpp)
set_target_properties(ninfer_deployment PROPERTIES POSITION_INDEPENDENT_CODE ON)
ninfer_internal_includes(ninfer_deployment)

if(NINFER_ENABLE_DISTRIBUTED)
  add_library(ninfer_distributed_wire STATIC wire.cpp)
  set_target_properties(ninfer_distributed_wire PROPERTIES POSITION_INDEPENDENT_CODE ON)
  ninfer_internal_includes(ninfer_distributed_wire)
  # Distributed model headers and their consumers must agree on rank-program layout.
  target_compile_definitions(ninfer_distributed_wire PUBLIC NINFER_ENABLE_DISTRIBUTED=1)
  target_link_libraries(ninfer_distributed_wire PUBLIC ninfer_deployment)
  add_library(ninfer_distributed_control STATIC
    transport.cpp launch.cpp startup.cpp bootstrap.cpp communication/preparation.cpp)
  ninfer_internal_includes(ninfer_distributed_control)
  target_link_libraries(ninfer_distributed_control PUBLIC
    ninfer_distributed_wire ninfer_cpu ninfer_distributed_cuda)
  add_library(ninfer_distributed_cuda SHARED
    communication.cpp communication/selection.cpp communication/nccl.cpp pipeline_link.cpp)
  ninfer_internal_includes(ninfer_distributed_cuda)
  target_link_libraries(ninfer_distributed_cuda PUBLIC ninfer_core ninfer_deployment PRIVATE ninfer::nccl)
  if(NINFER_TARGET STREQUAL "dgx_spark")
    pkg_check_modules(IBVERBS REQUIRED IMPORTED_TARGET libibverbs)
    target_sources(ninfer_distributed_cuda PRIVATE
      communication/native/backend.cpp communication/native/rdma.cpp
      communication/native/collective.cu)
    ninfer_cuda_library(ninfer_distributed_cuda)
    target_link_libraries(ninfer_distributed_cuda PRIVATE
      ninfer_cpu ninfer_distributed_wire PkgConfig::IBVERBS)
  endif()
endif()
