"""Small-file PLE preparation qualification; exact independent hash/embedding oracle."""

from pathlib import Path
from fractions import Fraction
import struct
import subprocess
import sys
import tempfile

from test_table import fixture, resegment, Artifact, exact_table

T, B, EOS = 5, 8, 248044
CONTROLS = (
    [-(1 << 63) + 23, 0x7123456789123456, -77] + [3] * 16 + [3 * h for h in range(16)]
)
VALID = [5, 4, 3, 2, 1, 0, 5, 3]
IDS = [[7 + b, EOS, 2147483647 - b, 3 + b, 21] for b in range(B)]
TAILS = [(12 + b, EOS if b % 2 == 0 else 4 + b) for b in range(B)]


def row_ids(ids, tail):
    result = []
    older, previous = tail
    mask = (1 << 64) - 1
    for current in ids:
        pair = (current * CONTROLS[0] & mask) ^ (previous * CONTROLS[1] & mask)
        triple = pair ^ ((EOS if previous == EOS else older) * CONTROLS[2] & mask)
        for h in range(16):
            word = pair if h < 8 else triple
            signed = word if word < (1 << 63) else word - (1 << 64)
            result.append(signed % CONTROLS[3 + h] + CONTROLS[19 + h])
        older, previous = previous, current
    return result


def reference(path):
    with Artifact(path) as artifact:
        decoded = []
        for i, divisor in enumerate((2, 3.25)):
            table = exact_table(Fraction(divisor))
            obj = artifact.by_id[f"shard-{i}"]
            raw = artifact.read_object(obj.id)
            for r in range(obj.shape[0]):
                begin = (r // 45) * 4096 + (r % 45) * 90
                row = raw[begin : begin + 90]
                decoded.append(
                    [
                        table[
                            row[80 + k // 16] * 16
                            + ((row[k // 2] >> (4 * (k % 2))) & 15)
                        ]
                        for k in range(160)
                    ]
                )
    hashes = [row_ids(ids, tail) for ids, tail in zip(IDS, TAILS)]
    embedding = []
    for b in range(B):
        for t in range(T):
            for h in range(16):
                embedding.extend(
                    decoded[hashes[b][t * 16 + h]] if t < VALID[b] else [0] * 160
                )
    return decoded, hashes, struct.pack(f"<{len(embedding)}H", *embedding)


def packet():
    result = struct.pack("<IIi", T, B, EOS)
    for b in range(B):
        result += struct.pack("<I2i5i", VALID[b], *TAILS[b], *IDS[b])
    return result


def run(executable, path, mode):
    result = subprocess.run(
        [executable, str(path), mode], input=packet(), stdout=subprocess.PIPE
    )
    if result.returncode == 77:
        raise SystemExit(77)
    assert result.returncode == 0, result.returncode
    return result.stdout


def main():
    executable, mode = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="ninfer-ple-prepare-") as directory:
        path = Path(directory) / "small.ninfer"
        fixture(path, hash_controls=CONTROLS)
        decoded, hashes, expected = reference(path)
        if mode == "--gpu":
            assert run(executable, path, mode) == expected * 2
        else:
            prefix = struct.pack("<160H", *decoded[44]) * 2
            prefix += struct.pack(f"<{B*T*16}I", *(x for row in hashes for x in row))
            assert run(executable, path, mode) == prefix + expected * 2
            split = Path(directory) / "split.ninfer"
            resegment(path, split)
            assert run(executable, split, mode) == prefix + expected * 2
            # Corrupt one requested row without growing the fixture or creating another artifact.
            with Artifact(path) as artifact:
                target = artifact.by_id["shard-0"].offset + 80
                for i, file in enumerate(artifact.directory.files):
                    if target < file.payload_bytes:
                        name = path if i == 0 else path.parent / file.path
                        offset = (artifact.payload_offset if i == 0 else 4096) + target
                        break
                    target -= file.payload_bytes
            with name.open("r+b") as f:
                f.seek(offset)
                f.write(b"\x7f")
            assert run(executable, path, "--failure") == b""
    print("PLE preparation: exact hash/E, lifecycle and resource handoff passed")


if __name__ == "__main__":
    main()
