#pragma once

#include "cache_reference.h"
#include "ops/op_tester.h"

namespace ninfer::test::qsa::sparse {

struct EncodedPlane {
    int leading, element;
    std::int64_t row, head, page;
    std::vector<std::uint8_t> host;
    GuardedDeviceBuffer device;
    Tensor view;

    EncodedPlane(int features, DType dtype, int pages, bool padded, int heads = 2)
        : leading(padded ? dtype_size(dtype) : 0), element(dtype_size(dtype)),
          row(features * element + (padded ? element : 0)), head(64 * row + leading),
          page(heads * head + leading), host(leading + page * pages, 0xcd), device(host.size()),
          view(device.span().subspan(leading), dtype, {features, 64, heads, pages}) {
        view.nb[1] = row;
        view.nb[2] = head;
        view.nb[3] = page;
    }

    std::size_t offset(int r, int h, int p) const {
        return leading + r * row + h * head + p * page;
    }

    void upload() { device.copy_from_host(host.data(), host.size()); }

    int preserved(const std::string& label) const {
        return verify_exact(label.c_str(), from_device<std::uint8_t>(device.data(), host.size()),
                            host) +
               device.verify_guards(label);
    }
};

struct EncodedVector {
    prepare::VectorEncoding format;
    EncodedPlane data, scales;

    EncodedVector(prepare::VectorEncoding fmt, int pages, bool padded, bool padded_scales = false,
                  int heads = 2)
        : format(fmt), data(fmt.data_extent, fmt.data_dtype, pages, padded, heads),
          scales(fmt.scale_extent, fmt.scale_dtype, pages, padded || padded_scales, heads) {}

    template <class Source>
    void encode(const Source& source) {
        std::fill(data.host.begin(), data.host.end(), 0xcd);
        std::fill(scales.host.begin(), scales.host.end(), 0xcd);
        for (int p = 0; p < data.view.ne[3]; ++p)
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
                        values[d]       = source.view.dtype == DType::BF16 ? bf16_to_f32(word)
                                                                           : prepare::fp16_decode(word);
                    }
                    const auto encoded = prepare::encode_row(values, format);
                    std::copy(encoded.data.begin(), encoded.data.end(),
                              data.host.begin() + data.offset(r, h, p));
                    std::copy(encoded.scales.begin(), encoded.scales.end(),
                              scales.host.begin() + scales.offset(r, h, p));
                }
        data.upload();
        scales.upload();
    }

    double decode(int d, int h, int r, int p) const {
        const auto db = data.offset(r, h, p), sb = scales.offset(r, h, p);
        const bool fp4  = format.encoding == prepare::Encoding::Nvfp4;
        const int group = d / format.group;
        const auto code = fp4 ? (data.host[db + d / 2] >> (4 * (d % 2))) & 15 : data.host[db + d];
        const double scale =
            fp4 ? prepare::positive_fp8(scales.host[sb + group])
                : prepare::fp16_decode(prepare::read_u16(scales.host, sb + 2 * group));
        return prepare::decode_code(code, format.encoding) * scale;
    }
};

} // namespace ninfer::test::qsa::sparse
