"""Persistent DFlash2 suffix shared by both Qwen3.8-27B weight profiles."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import BF16, TensorSpec, W8, tensor_spec


DFLASH2_LAYERS = tuple(range(5))


def _build_dflash2_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        tensor_spec("dflash2/feature_projection", (5120, 25600), W8),
        tensor_spec("dflash2/context_norm", (5120,), BF16),
    ]
    for layer in DFLASH2_LAYERS:
        prefix = f"dflash2/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "input_norm", (5120,), BF16),
                tensor_spec(
                    prefix + "attention_conv/base_kernel",
                    (2, 2, 5120),
                    BF16,
                ),
                tensor_spec(
                    prefix + "attention_conv/kernel_projection",
                    (1280, 5120),
                    BF16,
                ),
                tensor_spec(
                    prefix + "attention/query_key_value",
                    (6144, 5120),
                    W8,
                ),
                tensor_spec(prefix + "attention/query_norm", (128,), BF16),
                tensor_spec(prefix + "attention/key_norm", (128,), BF16),
                tensor_spec(
                    prefix + "attention/output",
                    (5120, 4096),
                    W8,
                ),
                tensor_spec(prefix + "post_attention_norm", (5120,), BF16),
                tensor_spec(
                    prefix + "mlp_conv/base_kernel",
                    (2, 2, 5120),
                    BF16,
                ),
                tensor_spec(
                    prefix + "mlp_conv/kernel_projection",
                    (1280, 5120),
                    BF16,
                ),
                tensor_spec(prefix + "mlp/gate_up", (34816, 5120), W8),
                tensor_spec(prefix + "mlp/down", (5120, 17408), W8),
            )
        )
    specs.extend(
        (
            tensor_spec("dflash2/final_norm", (5120,), BF16),
            tensor_spec(
                "dflash2/candidate_selector/hidden_projection",
                (256, 5120),
                BF16,
            ),
            tensor_spec(
                "dflash2/candidate_selector/predecessor_codebook",
                (248320, 256),
                BF16,
            ),
            tensor_spec(
                "dflash2/candidate_selector/successor_codebook",
                (248320, 256),
                BF16,
            ),
        )
    )
    return tuple(specs)


DFLASH2_TENSOR_SPECS = _build_dflash2_specs()


__all__ = [
    "DFLASH2_LAYERS",
    "DFLASH2_TENSOR_SPECS",
]
