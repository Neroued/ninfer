// Renames that turn upstream's NVFP4 linear_swiglu plan and A4 launcher into the [28672,5120]
// problem's (linear_swiglu_ffn14336.h) when a translation unit of this directory includes them.
// Include it once, after the family's headers and right before the upstream source.

#define Nvfp4N34816K5120              Nvfp4N28672K5120
#define nvfp4_linear_swiglu_a4_launch nvfp4_linear_swiglu_n28672_a4_launch
#define nvfp4_linear_swiglu_workspace_capacity_bytes                                               \
    nvfp4_linear_swiglu_n28672_workspace_capacity_bytes
#define nvfp4_linear_swiglu_dispatch nvfp4_linear_swiglu_n28672_dispatch
