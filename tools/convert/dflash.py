"""DFlash checkpoint mathematics and parameter mapping, independent of its target."""

from .config import expect_value, positive_float, positive_integer, rope_parameters


def draft_config(raw: dict, target: dict, backend: str) -> dict:
    architecture = "DFlash2DraftModel" if backend == "dflash2" else "DFlashDraftModel"
    expect_value(raw, "architectures", [architecture], backend)
    expect_value(raw, "model_type", "qwen3", backend)
    expect_value(raw, "hidden_act", "silu", backend)
    expect_value(raw, "attention_bias", False, backend)
    expect_value(raw, "is_causal", False, backend)
    for key, value in (
        ("hidden_size", target["hidden_size"]),
        ("vocab_size", target["vocab_size"]),
        ("num_target_layers", target["num_hidden_layers"]),
    ):
        expect_value(raw, key, value, backend)
    result = {"architectures": [architecture], "model_type": "qwen3"}
    for key in (
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "num_hidden_layers",
        "max_position_embeddings",
    ):
        result[key] = positive_integer(raw.get(key), backend + "." + key)
    if (
        result["num_attention_heads"] % result["num_key_value_heads"]
        or result["head_dim"] % 2
    ):
        raise ValueError(f"{backend}: invalid attention geometry")
    result["rms_norm_eps"] = positive_float(
        raw.get("rms_norm_eps", 1e-6), backend + ".rms_norm_eps"
    )
    rope = rope_parameters(raw, backend)
    result["rope_parameters"] = {
        "rope_theta": positive_float(
            rope.get("rope_theta", raw.get("rope_theta", 10_000_000)),
            backend + ".rope_theta",
        )
    }
    layers = raw.get("layer_types")
    if layers is None:
        enabled = raw.get("use_sliding_window", False)
        if type(enabled) is not bool:
            raise ValueError(f"{backend}.use_sliding_window must be boolean")
        # Qwen3Config applies sliding attention at and above this layer index.
        threshold = raw.get("max_window_layers", 28)
        if type(threshold) is not int or threshold < 0:
            raise ValueError(f"{backend}.max_window_layers must be nonnegative")
        layers = [
            "sliding_attention" if enabled and i >= threshold else "full_attention"
            for i in range(result["num_hidden_layers"])
        ]
    if (
        not isinstance(layers, list)
        or len(layers) != result["num_hidden_layers"]
        or any(k not in ("sliding_attention", "full_attention") for k in layers)
    ):
        raise ValueError(f"{backend}: invalid layer_types")
    result["layer_types"] = list(layers)
    if "sliding_attention" in layers:
        result["sliding_window"] = positive_integer(
            raw.get("sliding_window"), backend + ".sliding_window"
        )
    draft = raw.get("dflash_config", {})
    taps = draft.get("target_layer_ids")
    if (
        not isinstance(taps, list)
        or not taps
        or any(
            type(i) is not int or not 0 <= i < target["num_hidden_layers"] for i in taps
        )
        or len(set(taps)) != len(taps)
    ):
        raise ValueError(f"{backend}: invalid target_layer_ids")
    mask = draft.get("mask_token_id")
    if type(mask) is not int or not 0 <= mask < target["vocab_size"]:
        raise ValueError(f"{backend}: mask token outside target embedding")
    output = {"target_layer_ids": list(taps), "mask_token_id": mask}
    if "block_size" in draft:
        output["block_size"] = positive_integer(
            draft["block_size"], backend + ".block_size"
        )
    if backend == "dflash2":
        for key in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            output[key] = positive_integer(draft.get(key), backend + "." + key)
        if target["hidden_size"] % output["conv_group_size"]:
            raise ValueError(
                "DFlash2 hidden width must be divisible by conv_group_size"
            )
        for scope in (raw, draft):
            for key, expected in (
                ("sample_from_anchor", False),
                ("input_embedding_scale", 1.0),
                ("output_multiplier", 1.0),
            ):
                expect_value(scope, key, expected, backend)
            if scope.get("final_logit_softcapping") not in (None, 0, 0.0):
                raise ValueError("DFlash2 final_logit_softcapping must be disabled")
    result["dflash_config"] = output
    return result


def add_draft(model, add, store, config, target, backend):
    h, d = target["hidden_size"], config["head_dim"]
    q, k = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    draft = config["dflash_config"]
    add(
        backend + "/feature_projection",
        store,
        "fc.weight",
        (h, len(draft["target_layer_ids"]) * h),
        inputs=(backend + "/target_features",),
    )
    add(backend + "/context_norm", store, "hidden_norm.weight", (h,))
    add(backend + "/final_norm", store, "norm.weight", (h,))
    for i in range(config["num_hidden_layers"]):
        p, sp = f"{backend}/layers/{i}/", f"layers.{i}."
        add(p + "input_norm", store, sp + "input_layernorm.weight", (h,))
        add(
            p + "post_attention_norm",
            store,
            sp + "post_attention_layernorm.weight",
            (h,),
        )
        for role, field, rows in (
            ("query", "q_proj", q),
            ("key", "k_proj", k),
            ("value", "v_proj", k),
        ):
            add(
                p + "attention/" + role,
                store,
                sp + "self_attn." + field + ".weight",
                (rows, h),
                inputs=(p + "query_projection_input",),
            )
            if role in ("key", "value"):
                add(
                    p + "attention/context_" + role,
                    store,
                    sp + "self_attn." + field + ".weight",
                    (rows, h),
                    inputs=(backend + "/context_input",),
                )
        model.packing_groups.append(
            tuple(p + "attention/" + r for r in ("query", "key", "value"))
        )
        for role, field in (("query_norm", "q_norm"), ("key_norm", "k_norm")):
            add(
                p + "attention/" + role,
                store,
                sp + "self_attn." + field + ".weight",
                (d,),
            )
        add(
            p + "attention/output",
            store,
            sp + "self_attn.o_proj.weight",
            (h, q),
            inputs=(p + "attention_output",),
        )
        intermediate = config["intermediate_size"]
        for role in ("gate", "up"):
            add(
                p + "mlp/" + role,
                store,
                sp + "mlp." + role + "_proj.weight",
                (intermediate, h),
                inputs=(p + "mlp_input",),
            )
        add(
            p + "mlp/down",
            store,
            sp + "mlp.down_proj.weight",
            (h, intermediate),
            inputs=(p + "mlp_product",),
        )
        model.packing_groups.append((p + "mlp/gate", p + "mlp/up"))
        if backend == "dflash2":
            taps, groups = draft["conv_kernel_size"], h // draft["conv_group_size"]
            for branch in ("attention_conv", "mlp_conv"):
                add(
                    p + branch + "/base_kernel",
                    store,
                    sp + branch + ".base_kernel",
                    (2, taps, h),
                )
                add(
                    p + branch + "/kernel_projection",
                    store,
                    sp + branch + ".kernel_projection.weight",
                    (2 * taps * groups, h),
                    inputs=(p + branch + "_input",),
                )
    if backend == "dflash2":
        rank = draft["selector_rank"]
        add(
            backend + "/candidate_selector/hidden_projection",
            store,
            "candidate_selector.hidden_projection.weight",
            (rank, h),
            inputs=(backend + "/final_hidden",),
        )
        for role in ("predecessor_codebook", "successor_codebook"):
            add(
                backend + "/candidate_selector/" + role,
                store,
                "candidate_selector." + role,
                (target["vocab_size"], rank),
            )
