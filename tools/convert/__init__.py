"""Offline checkpoint-to-artifact conversion.

sources reads and interprets checkpoint values. qwen3_5 maps those values and
configuration to model's logical parameters; resources supplies frontend bytes.
recipe selects sources, representations, grouping, and methods. official_recipes
supplies the maintained assignments; proposal adds the indexed proposal head.
methods adapts numerical quantization or encoded import to artifact.tensor_output.
pipeline prepares jobs, runs them through the artifact writer, and reports results.
__main__ supplies the CLI and loads explicitly selected recipe functions.
"""
