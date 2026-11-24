"""Numerical algorithms producing codes and stored scales from logical values.

groupwise covers Q4/Q5/Q6/Q8 using one parameterized algorithm; fp8_row and
fp8_block share the BF16-scale FP8 numerical rules across reduction domains.
Exact byte packing belongs to tools.artifact.codecs.
"""
