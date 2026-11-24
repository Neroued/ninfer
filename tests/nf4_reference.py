"""Exact test-owned reconstruction, independent of production codec arithmetic."""

import bisect
from fractions import Fraction
import struct


def fp32(word):
    return Fraction(struct.unpack("<f", struct.pack("<I", word))[0])


code_words = [
    0xBF800000,
    0xBF3239B1,
    0xBF066B30,
    0xBECA32A0,
    0xBE91A24D,
    0xBE3D353F,
    0xBDBA7871,
    0x00000000,
    0x3DA2FAFF,
    0x3E24CAE3,
    0x3E7C04DD,
    0x3EAD033A,
    0x3EE1A4B8,
    0x3F1007AB,
    0x3F3913B3,
    0x3F800000,
]
codes = [fp32(w) for w in code_words]
bf16 = [fp32(w << 16) for w in range(0x7F80)]
overflow = bf16[-1] + (bf16[-1] - bf16[-2]) / 2
scales = [
    (
        Fraction(s, 512)
        if s < 8
        else Fraction(8 + (s & 7)) * Fraction(2) ** ((s >> 3) - 10)
    )
    for s in range(127)
]


def rounded(value):
    pos = bisect.bisect_left(bf16, value)
    return min(
        range(max(0, pos - 1), min(len(bf16), pos + 1)),
        key=lambda w: (abs(bf16[w] - value), w & 1),
    )


def table(divisor):
    return [
        ((code_words[c] >> 16) & 0x8000) | rounded(abs(code * scale / divisor))
        for scale in scales
        for c, code in enumerate(codes)
    ] + [0] * 16
