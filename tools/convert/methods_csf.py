"""Stream native CSF planes into parent matrices without expanding scale words."""

from __future__ import annotations

import numpy as np

from tools.artifact.layouts import csf32_geometry
from .methods import AuxiliaryValue, CHUNK_BYTES, PrepareRequest, _preflight


def _exception_counts(source):
    planes = source.native_planes
    geometry = csf32_geometry(source.shape, planes.exception_count)
    offsets = np.frombuffer(
        planes.read("offsets", 0, planes.sizes["offsets"]).numpy(), dtype="<u2"
    )
    bases = np.frombuffer(
        planes.read("page_bases", 0, planes.sizes["page_bases"]).numpy(), dtype="<u4"
    )
    counts = np.empty(geometry.blocks_per_matrix, dtype=np.uint32)
    cursor = 0
    for page, first in enumerate(range(0, geometry.blocks_per_matrix, 256)):
        count = min(256, geometry.blocks_per_matrix - first)
        local = offsets[first + page : first + page + count + 1].astype(np.uint32)
        if local[0] != 0 or bases[page] != cursor * 2 or np.any(local[1:] < local[:-1]):
            raise ValueError(f"{source.label}: invalid CSF exception partition")
        deltas = np.diff(local)
        if np.any(deltas > 64):
            raise ValueError(f"{source.label}: CSF block has too many exceptions")
        counts[first : first + count] = deltas
        cursor += int(local[-1])
    if cursor != planes.exception_count:
        raise ValueError(
            f"{source.label}: CSF exception count differs from its offsets"
        )
    return counts


def import_csf(request: PrepareRequest):
    _preflight(request, values=False)
    if request.target.format != "nvfp4_mul" or len(request.target.shape) not in (2, 3):
        raise ValueError("native CSF import requires NVFP4 multiplier matrix/bank")
    n, k = request.target.shape[-2:]
    geometry = csf32_geometry(
        request.target.shape, request.target.layout_parameters["exception_count"]
    )
    matrices, rows, current = [], 0, []
    auxiliaries = {}
    for item in request.inputs:
        source = item.source
        planes = source.native_planes
        if (
            len(source.shape) != 2
            or source.shape[1] != k
            or source.shape[0] % 128
            or k % 64
        ):
            raise ValueError(
                "native CSF import needs complete 128-row, K64-aligned matrices"
            )
        if (
            planes is None
            or planes.layout != "csf_32x2_v1"
            or source.encoding() != "nvfp4_mul"
        ):
            raise ValueError(f"{item.parameter}: native CSF planes are unavailable")
        expected = csf32_geometry(source.shape, planes.exception_count)
        if planes.sizes != {name: size for name, _, size in expected.planes}:
            raise ValueError(
                f"{item.parameter}: native CSF plane sizes differ from geometry"
            )
        if source.weight_scale is None:
            raise ValueError("native CSF needs its weight multiplier")
        multiplier = source.weight_scale()
        if current and multiplier != current[0].weight_scale():
            raise ValueError("NVFP4 multipliers differ within one parent matrix")
        current.append(source)
        rows += source.shape[0]
        if rows > n:
            raise ValueError("native CSF source crosses a parent matrix boundary")
        if rows == n:
            matrices.append(tuple(current))
            current, rows = [], 0
        for parameter, input_name in item.uses:
            if request.policies[(parameter, input_name)] != "AllowA4":
                continue
            key = (parameter, input_name, "activation_input_multiplier")
            value = request.auxiliary_overrides.get(key)
            if value is None:
                if source.activation_scale is None:
                    raise ValueError(
                        f"{parameter}: missing native activation multiplier"
                    )
                value = AuxiliaryValue.activation_scale(source.activation_scale())
            if key in auxiliaries and auxiliaries[key] != value:
                raise ValueError(
                    f"{parameter}: supply an expert-indexed activation multiplier range"
                )
            auxiliaries[key] = value
    if current or len(matrices) != geometry.matrices:
        raise ValueError("native CSF inputs do not cover the parent matrices")

    def produce(output):
        exception_cursor = 0
        for matrix, sources in enumerate(matrices):
            row, counts = 0, []
            matrix_exception_begin = exception_cursor
            for source in sources:
                planes = source.native_planes
                counts.append(_exception_counts(source))
                for name, destination in (
                    ("codes", (matrix * n + row) * k // 2),
                    ("bases", matrix * n + row),
                    ("nibbles", (matrix * n + row) * (k // 16) // 2),
                    ("exceptions", exception_cursor * 2),
                ):
                    read = lambda begin, end, name=name: planes.read(name, begin, end)
                    for begin, _, data in request.read_chunks(
                        read, planes.sizes[name], CHUNK_BYTES
                    ):
                        output.write_csf_plane(
                            name,
                            destination + begin,
                            memoryview(data.numpy()).cast("B"),
                        )
                row += source.shape[0]
                exception_cursor += planes.exception_count
            counts = np.concatenate(counts)
            offsets = np.empty(geometry.offsets_per_matrix, dtype="<u2")
            page_bases = np.empty(geometry.pages_per_matrix, dtype="<u4")
            cursor = matrix_exception_begin
            for page, first in enumerate(range(0, geometry.blocks_per_matrix, 256)):
                local = np.cumsum(counts[first : first + 256], dtype=np.uint32)
                page_bases[page] = cursor * 2
                offsets[first + page] = 0
                offsets[first + page + 1 : first + page + 1 + len(local)] = local
                cursor += int(local[-1])
            output.write_csf_plane(
                "offsets",
                matrix * geometry.offsets_per_matrix * 2,
                memoryview(offsets).cast("B"),
            )
            output.write_csf_plane(
                "page_bases",
                matrix * geometry.pages_per_matrix * 4,
                memoryview(page_bases).cast("B"),
            )
            output.write_csf_plane(
                "matrix_scale", matrix * 4, sources[0].weight_scale()
            )
        if exception_cursor != geometry.exception_count:
            raise ValueError(
                "native CSF exception payload does not cover its declared extent"
            )

    return request.job(produce=produce, auxiliaries=auxiliaries)
