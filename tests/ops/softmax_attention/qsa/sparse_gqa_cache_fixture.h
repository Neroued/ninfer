#pragma once

#include "cache_reference.h"
#include "ops/op_tester.h"
#include "ops/suite.h"
#include <cuda.h>
#include "../../../../tools/support/paged_kv_fixture.h"

namespace ninfer::test::qsa::sparse {

template <class T>
void upload_pages(tool::PackedKVFixture& packed, std::size_t plane_index, std::span<const T> host,
                  std::span<const int> dirty_pages) {
    const auto& region = packed.layout().planes[plane_index].tensor.region;
    const auto width   = region.bytes;
    auto* base         = static_cast<std::byte*>(packed.storage().data) + region.offset;
    for (std::size_t first = 0; first < dirty_pages.size();) {
        std::size_t last = first + 1;
        while (last < dirty_pages.size() && dirty_pages[last] == dirty_pages[last - 1] + 1) ++last;
        const int page = dirty_pages[first];
        CUDA_CHECK(cudaMemcpy2D(base + packed.physical_offset(page), packed.record_stride(),
                                host.data() + std::size_t(page) * (width / sizeof(T)), width, width,
                                last - first, cudaMemcpyHostToDevice));
        first = last;
    }
}

template <class T>
void initialize_plane(tool::PackedKVFixture& packed, std::size_t plane_index, T poison, int pages) {
    const auto& region = packed.layout().planes[plane_index].tensor.region;
    auto* base =
        static_cast<std::byte*>(packed.storage().data) + region.offset + packed.physical_offset(0);
    if constexpr (sizeof(T) == 1) {
        CUDA_CHECK(cudaMemset2D(base, packed.record_stride(), poison, region.bytes, pages));
    } else {
        if (cuMemsetD2D16(reinterpret_cast<CUdeviceptr>(base), packed.record_stride(), poison,
                          region.bytes / sizeof(T), pages) != CUDA_SUCCESS)
            throw std::runtime_error("initialize sparse QSA plane poison");
    }
}

struct EncodedPlane {
    int element;
    std::int64_t row, head, page;
    std::vector<std::uint8_t> host;
    tool::PackedKVFixture* packed = nullptr;
    std::size_t plane_index       = 0;
    Tensor view;

    EncodedPlane(int features, DType dtype, int pages, int heads = 2)
        : element(dtype_size(dtype)), row(features * element), head(64 * row), page(heads * head),
          host(page * pages, 0xcd), view(Tensor::metadata(dtype, {features, 64, heads, pages})) {
        view.nb[1] = row;
        view.nb[2] = head;
        view.nb[3] = page;
    }

    std::size_t offset(int r, int h, int p) const { return r * row + h * head + p * page; }

    void bind(tool::PackedKVFixture& arena, std::size_t index) {
        packed      = &arena;
        plane_index = index;
        initialize_plane(arena, index, std::uint8_t{0xcd}, view.ne[3]);
    }

    void upload() {
        auto source   = view;
        source.data   = host.data();
        source.device = Device::cpu();
        packed->upload_plane(plane_index, source);
    }

    int preserved(const std::string& label) const {
        if (packed) {
            auto actual   = host;
            auto target   = view;
            target.data   = actual.data();
            target.device = Device::cpu();
            packed->download_plane(plane_index, target);
            return verify_exact(label.c_str(), actual, host);
        }
        throw std::logic_error("encoded plane is not bound");
    }
};

struct EncodedVector {
    prepare::VectorEncoding format;
    EncodedPlane data, scales;

    EncodedVector(prepare::VectorEncoding fmt, int pages, int heads = 2)
        : format(fmt), data(fmt.data_extent, fmt.data_dtype, pages, heads),
          scales(fmt.scale_extent, fmt.scale_dtype, pages, heads) {}

    template <class Source>
    void encode(const Source& source, std::span<const int> dirty_pages) {
        parallel_rows(
            dirty_pages.size(), std::uint64_t(dirty_pages.size()) * data.page,
            [&](int first, int last) {
                for (int i = first; i < last; ++i) {
                    const int p = dirty_pages[i];
                    std::fill_n(data.host.data() + p * data.page, data.page, 0xcd);
                    std::fill_n(scales.host.data() + p * scales.page, scales.page, 0xcd);
                    for (int h = 0; h < data.view.ne[2]; ++h)
                        for (int r = 0; r < 64; ++r) {
                            const auto first   = source.host[source.offset(0, h, r, p)];
                            const auto initial = source.view.dtype == DType::BF16
                                                     ? bf16_to_f32(first)
                                                     : prepare::fp16_decode(first);
                            if (!std::isfinite(initial)) continue;
                            std::array<float, 256> values;
                            for (int d = 0; d < 256; ++d) {
                                const auto word = source.host[source.offset(d, h, r, p)];
                                values[d]       = source.view.dtype == DType::BF16
                                                      ? bf16_to_f32(word)
                                                      : prepare::fp16_decode(word);
                            }
                            auto codes =
                                std::span(data.host).subspan(data.offset(r, h, p), data.row);
                            auto exponents =
                                std::span(scales.host).subspan(scales.offset(r, h, p), scales.row);
                            prepare::encode_row_into(values, format, codes, exponents);
                            // Exercise independently scaled groups in every reader route.
                            // The attention oracle consumes these actual stored operands.
                            if (format.encoding == prepare::Encoding::Fp8)
                                exponents[(r + h) % 2] -= 3;
                        }
                }
            });
        upload_pages(*data.packed, data.plane_index, std::span<const std::uint8_t>(data.host),
                     dirty_pages);
        upload_pages(*scales.packed, scales.plane_index, std::span<const std::uint8_t>(scales.host),
                     dirty_pages);
    }

    void decode_row(int h, int r, int p, double* decoded) const {
        const auto* codes = data.host.data() + data.offset(r, h, p);
        const auto sb     = scales.offset(r, h, p);
        const bool fp4    = format.encoding == prepare::Encoding::Nvfp4;
        for (int begin = 0; begin < 256; begin += format.group) {
            const int group = begin / format.group;
            const double scale =
                fp4 ? prepare::positive_fp8(scales.host[sb + group])
                : format.encoding == prepare::Encoding::Fp8
                    ? fp8_cache_multiplier(scales.host[sb + group])
                    : prepare::fp16_decode(prepare::read_u16(scales.host, sb + 2 * group));
            for (int d = begin; d < begin + format.group; ++d) {
                const auto code = fp4 ? (codes[d / 2] >> (4 * (d % 2))) & 15 : codes[d];
                decoded[d]      = prepare::decode_code(code, format.encoding) * scale;
            }
        }
    }
};

} // namespace ninfer::test::qsa::sparse
