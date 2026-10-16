set(NINFER_TARGET rtx5090 CACHE STRING "NInfer execution target")
set_property(CACHE NINFER_TARGET PROPERTY STRINGS rtx5090 dgx_spark)

if(DEFINED NINFER_CONFIGURED_TARGET AND
   NOT NINFER_CONFIGURED_TARGET STREQUAL NINFER_TARGET)
  message(FATAL_ERROR
    "This build directory belongs to ${NINFER_CONFIGURED_TARGET}; "
    "use a separate directory for ${NINFER_TARGET}")
endif()

if(NINFER_TARGET STREQUAL "rtx5090")
  set(NINFER_CUDA_ARCHITECTURE 120a)
  set(NINFER_SHARED_PHYSICAL_MEMORY 0)
  set(NINFER_PROCESSOR_PATTERN "^(x86_64|amd64|AMD64)$")
elseif(NINFER_TARGET STREQUAL "dgx_spark")
  set(NINFER_CUDA_ARCHITECTURE 121a)
  set(NINFER_SHARED_PHYSICAL_MEMORY 1)
  set(NINFER_PROCESSOR_PATTERN "^(aarch64|arm64|ARM64)$")
else()
  message(FATAL_ERROR "Unknown NINFER_TARGET '${NINFER_TARGET}'")
endif()

# CUDA compiler detection must see the selected architecture, not a host default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${NINFER_CUDA_ARCHITECTURE}
    CACHE STRING "CUDA architectures to build")
endif()
function(ninfer_validate_cuda_architecture)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL NINFER_CUDA_ARCHITECTURE)
    message(FATAL_ERROR
      "${NINFER_TARGET} requires CMAKE_CUDA_ARCHITECTURES=${NINFER_CUDA_ARCHITECTURE}; "
      "got '${CMAKE_CUDA_ARCHITECTURES}'")
  endif()
endfunction()
ninfer_validate_cuda_architecture()

function(ninfer_validate_platform)
  # The toolchain file can change CUDA architectures during project() initialization.
  ninfer_validate_cuda_architecture()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "${NINFER_PROCESSOR_PATTERN}")
    message(FATAL_ERROR
      "${NINFER_TARGET} does not support ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  set(NINFER_CONFIGURED_TARGET ${NINFER_TARGET} CACHE INTERNAL "Configured NInfer target")
  message(STATUS "NInfer target: ${NINFER_TARGET} (${CMAKE_SYSTEM_PROCESSOR}, sm_${NINFER_CUDA_ARCHITECTURE})")
endfunction()
