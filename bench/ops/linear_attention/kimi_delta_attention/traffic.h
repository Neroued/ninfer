#pragma once
#include "fixture.cuh"
#include "common/roofline.h"
#include <numeric>
#include <sstream>

namespace ninfer::bench::kda {
struct Traffic {
    std::uint64_t read = 0, write = 0, state_read = 0, state_write = 0, record_read = 0,
                  record_write = 0;
    double projection_flops    = 0;

    std::string csv(double us) const {
        const auto bytes = read + write;
        std::ostringstream s;
        s << read << ',' << write << ',' << state_read << ',' << state_write << ',' << record_read
          << ',' << record_write << ',' << bytes / (us * 1000) << ','
          << kRoofline.sustained_read_gbs << ',' << bytes / (kRoofline.sustained_read_gbs * 1000)
          << ',' << projection_flops << ',';
        if (projection_flops) s << projection_flops / (kRoofline.bf16_f32_tflops * 1e6);
        return s.str();
    }

    static constexpr const char* columns =
        "logical_read_bytes,logical_write_bytes,state_read_bytes,state_write_bytes,record_read_"
        "bytes,record_write_bytes,logical_GBps,"
        "memory_roofline_GBps,streaming_memory_reference_us,projection_flops,bf16_projection_"
        "reference_us";
};

inline Traffic traffic(Fixture& f, const std::string& op) {
    const auto n = std::accumulate(f.valid_counts.begin(), f.valid_counts.end(), std::uint64_t(0));
    const std::uint64_t state         = 4ULL * 128 * 128 * f.h * f.b;
    const std::uint64_t record_column = 2ULL * 128 * f.qh + 6ULL * 128 * f.h + 4ULL * f.h;
    Traffic out;
    if (op == "fold") {
        for (int m : f.commit_counts)
            if (m) {
                const auto bytes = 4ULL * 128 * 128 * f.h * f.layers;
                out.state_read += bytes + 2ULL * f.channels * f.layers * std::max(0, 3 - m);
                out.state_write += bytes + 6ULL * f.channels * f.layers;
                out.record_read +=
                    record_column * f.layers * m + 2ULL * f.channels * f.layers * std::min(m, 3);
                out.read += bytes + record_column * f.layers * m + 6ULL * f.channels * f.layers +
                            2 * sizeof(std::int64_t);
                out.write += bytes + 6ULL * f.channels * f.layers;
            }
        out.read += f.b * sizeof(std::int32_t);
        return out;
    }
    if (op == "input-update" || op == "input-record") {
        out.state_read   = 6ULL * f.channels * f.b;
        out.state_write  = op == "input-update" ? out.state_read : 0;
        out.record_write = op == "input-record" ? 2ULL * f.channels * n : 0;
        out.read = f.weight.model_weight_bytes() + 2ULL * 4096 * f.t * f.b + 16ULL * f.channels +
                   out.state_read;
        out.write            = 2ULL * f.channels * f.t * f.b + out.state_write + out.record_write;
        out.projection_flops = 2.0 * 12288 * 4096 * f.t * f.b;
    } else if (op == "chain") {
        auto input = traffic(f, "input-record"), rec = traffic(f, "record"),
             fold = traffic(f, "fold");
        // Sum per-Op I/O; intermediate records/outputs intentionally count producer and consumer.
        out.read             = input.read + rec.read + fold.read;
        out.write            = input.write + rec.write + fold.write;
        out.state_read       = input.state_read + rec.state_read + fold.state_read;
        out.state_write      = fold.state_write;
        out.record_read      = fold.record_read;
        out.record_write     = input.record_write + rec.record_write;
        out.projection_flops = input.projection_flops;
    } else {
        out.state_read    = state;
        out.state_write   = op == "record" ? 0 : state;
        const auto copied = f.aliases_kv_records() ? 0 : 2ULL * 128 * (f.qh + f.h);
        out.record_write  = op == "record" ? (copied + 4ULL * 129 * f.h) * n : 0;
        out.read = (4ULL * 128 * f.qh + 4ULL * 128 * f.h + 2ULL * f.h) * n + 4ULL * 129 * f.h +
                   out.state_read;
        out.write = 2ULL * 128 * f.h * f.t * f.b + out.state_write + out.record_write;
    }
    return out;
}
} // namespace ninfer::bench::kda
