"""Independent Fraction oracle for the production C++ NF4-to-BF16 table."""

from fractions import Fraction
import random
import struct
import subprocess
import sys

from nf4_reference import code_words, codes, fp32, overflow, rounded, scales

rng = random.Random(20261013)
divisors = [
    0,
    1,
    0x80000000,
    0x7F800000,
    0x7FC00000,
    0xBF800000,
    0x3F800000,
    0x40000000,
    0x40500000,
    0x7F7FFFFF,
    0x03800000,
    0x03E00000,
    0x03E00001,
]
divisors += [rng.randrange(0x00800000, 0x7F800000) for _ in range(80)]
# Midpoints near one and divisors from the PLE scale range.
divisors += [0x3F800000 + delta for delta in (-1, 1, 32767, 32768, 32769, 65535)]
divisors += [
    struct.unpack("<I", struct.pack("<f", 448 / (v / 65536)))[0]
    for v in (3008, 4096, 5856)
]
data = subprocess.run(
    [sys.argv[1], "--tables"],
    input=struct.pack(f"<{len(divisors)}I", *divisors),
    check=True,
    stdout=subprocess.PIPE,
).stdout
offset = 0
checked = 0
for word in divisors:
    valid = word != 0 and word < 0x7F800000
    d = fp32(word) if valid else None
    valid = valid and Fraction(448) / d < overflow
    (got_valid,) = struct.unpack_from("<I", data, offset)
    offset += 4
    assert got_valid == valid, hex(word)
    if not valid:
        continue
    got = struct.unpack_from("<2048H", data, offset)
    offset += 4096
    for s, scale in enumerate(scales):
        for c, code in enumerate(codes):
            sign = (code_words[c] >> 16) & 0x8000
            expected = sign | rounded(abs(code * scale / d))
            assert got[s * 16 + c] == expected, (
                hex(word),
                s,
                c,
                got[s * 16 + c],
                expected,
            )
            checked += 1
    assert not any(got[127 * 16 :])
assert offset == len(data)
print(f"NF4 exact BF16 oracle: {checked} entries passed")
