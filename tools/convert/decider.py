"""Recipe for pplx-decider checkpoints: self-quantized NVFP4 text backbone.

The checkpoint ships BF16 weights and no language-model head, so projections are
quantized here with the weight divisor chosen per packed parent. Activations stay
16-bit; there is no calibration data for activation divisors.
"""

from __future__ import annotations

from .methods import fp8_row_maxabs, nvfp4_absmax


def pplx_decider_nvfp4(model, recipe, sources):
    if "decision" not in model.components:
        raise ValueError("this recipe requires a decision checkpoint")
    recipe.assign("text/token_embedding", format="fp8_e4m3fn_row_bf16", method=fp8_row_maxabs)
    # Decision scoring never reads the language-model head; alias it to the embedding.
    recipe.share("text/output_head", "text/token_embedding")
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        recipe.assign(name, format="nvfp4", method=nvfp4_absmax)


def pplx_decider_groupwise(model, recipe, sources):
    """Q4/Q5 groupwise backbone, usable at every prefill length without calibration.

    The engine registers its fused NVFP4 MLP and GDN input projections for 16-bit
    activations only through T=16, so a one-shot prompt prefill needs either
    calibrated NVFP4 activation divisors or a groupwise representation.
    """
    from .official_recipes import qwen3_8_27b

    if "decision" not in model.components:
        raise ValueError("this recipe requires a decision checkpoint")
    qwen3_8_27b(model, recipe, sources)
    recipe.share("text/output_head", "text/token_embedding")
