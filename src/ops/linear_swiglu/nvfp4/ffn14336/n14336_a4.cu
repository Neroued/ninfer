// The [14336,5120] gate|up problem's copy of upstream's nvfp4/nvfp4_linear_swiglu_a4.cu: the same
// source, compiled with the problem's geometry and entry points (linear_swiglu_ffn14336.h).
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/linear_swiglu/nvfp4/ffn14336/linear_swiglu_ffn14336.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "ops/linear_swiglu/nvfp4/ffn14336/n14336_names.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu"
