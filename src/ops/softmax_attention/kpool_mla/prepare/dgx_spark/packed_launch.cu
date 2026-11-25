// Keep the packed kernel ABI in its own translation unit so CUDA code generation
// for the rectangular Graph kernels retains the original shared-memory layout.
#define NINFER_KPOOL_PACKED
#include "ops/softmax_attention/kpool_mla/prepare/dgx_spark/launch_impl.cuh"
