#include "ninfer/ops/short_conv.h"
#include "op_measurement.cuh"
#include "ragged_workload.h"

#include <memory>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct Options {
    measurement::Options timing;
    std::vector<ragged::SegmentSpec> segments;
    std::vector<int> widths{1};
    int d = 2304, width = 1, batch = 1, stacked = 3, q = 1;
    bool snapshot = false, norm = true, conv_add = true, res = true, attn_add = true;
    ragged::StateOffsetForm state_form = ragged::StateOffsetForm::Distinct;
    std::string mode                   = "all";
};

void run(const Options& o, const ragged::SegmentSpec& lengths,
         measurement::LogicalBenchmark& benchmark) {
    auto& context     = device_context();
    const auto device = context.device();
    const auto packed =
        lengths.empty() ? nullptr : std::make_unique<ragged::RaggedDeviceFixture>(lengths, device);
    const auto view = packed ? packed->view() : ops::RaggedSequenceView{};
    const int rows  = packed ? view.num_seqs() : o.batch;
    const int batch = packed ? 1 : rows, width = packed ? view.total_tokens() : o.width;
    if (std::int64_t(width) * batch > std::numeric_limits<int>::max() ||
        (o.snapshot && std::int64_t(width) * batch + rows > std::numeric_limits<int>::max()))
        throw std::invalid_argument("too many batched columns");
    if (o.snapshot && (packed || o.state_form != ragged::StateOffsetForm::Distinct))
        throw std::invalid_argument("snapshot benchmark requires a distinct batched destination");
    const int m = width * batch, d = o.d, q = o.q;
    const auto count = std::size_t(d) * m;
    const auto shape = {d, width, batch};
    auto x = make_bf16(count, 511), w = make_bf16(std::size_t(d) * 4, 512);
    auto residual = o.res && o.conv_add ? make_bf16(count, 513) : DeviceBuffer{};
    auto conv_add = o.conv_add ? make_bf16(count, 514) : DeviceBuffer{};
    auto conv     = make_buffer(count * 2);
    const Tensor tx(x.span(), DType::BF16, shape), tw(w.span(), DType::BF16, {d, 4});
    const auto rr = residual.size() ? Tensor(residual.span(), DType::BF16, shape) : Tensor{};
    const auto ca = o.conv_add ? Tensor(conv_add.span(), DType::BF16, shape) : Tensor{};
    Tensor co(conv.span(), DType::BF16, shape);

    const auto state_bytes = std::size_t(d) * 3 * 2;
    const auto offsets     = ragged::make_state_offsets(rows, o.state_form, state_bytes, 16);
    auto history           = make_bf16(offsets.backing_bytes() / 2, 519);
    auto so = ragged::upload(device, std::span<const std::int64_t>(offsets.source_offsets_host));
    auto eo =
        ragged::upload(device, std::span<const std::int64_t>(offsets.destination_offsets_host));
    const TensorRegion region{{0, state_bytes, 16}, DType::BF16, {d, 3, 1, 1}};
    const StateFieldLayout field{region, 1, 0};
    const StateFieldView source{{history.span(), Tensor(so.span(), DType::I64, {rows})}, field};
    const StateFieldView dest{{history.span(), Tensor(eo.span(), DType::I64, {rows})}, field};
    std::unique_ptr<SavedBuffer> saved;
    if (offsets.mutates_source) saved = std::make_unique<SavedBuffer>(history);
    const auto restore = [&](CudaExecutionView cuda) {
        if (saved) saved->restore(cuda);
    };

    DeviceBuffer snapshot_storage, slots_storage;
    Tensor snapshots, slots;
    if (o.snapshot) {
        std::vector<int> bases(rows);
        for (int b = 0; b < rows; ++b) bases[b] = b * (width + 1);
        snapshot_storage = make_buffer(std::size_t(m + rows) * state_bytes);
        slots_storage    = ragged::upload(device, std::span<const int>(bases));
        snapshots        = Tensor(snapshot_storage.span(), DType::BF16, {d, 3, m + rows});
        slots            = Tensor(slots_storage.span(), DType::I32, {rows});
    }
    const auto standalone = [&](CudaExecutionView cuda) {
        if (packed)
            ops::short_conv_residual(tx, tw, view, source, dest, rr, ca, co, cuda);
        else if (o.snapshot)
            ops::short_conv_residual_snapshot(tx, tw, source, snapshots, {}, slots, rr, ca, co,
                                              cuda);
        else
            ops::short_conv_residual(tx, tw, source, dest, {}, rr, ca, co, cuda);
    };
    const auto state_read   = state_bytes * offsets.num_source_records;
    const auto state_write  = state_bytes * (o.snapshot ? m : rows);
    const auto metadata     = packed ? (std::size_t(m) + rows + 1) * 4 : 0;
    const auto offset_bytes = std::size_t(rows) * (8 + (o.snapshot ? 4 : 8));
    const auto conv_bytes = x.size() + w.size() + residual.size() + conv_add.size() + conv.size() +
                            state_read + state_write + metadata + offset_bytes;
    measurement::Row row;
    row.format        = "bf16";
    row.policy        = ragged::state_form_name(o.state_form);
    row.conv_channels = d;
    row.tokens        = m;
    row.batch         = rows;
    row.valid         = packed ? ragged::format_segments(lengths) : std::to_string(width);
    row.commits       = packed ? "ragged" : o.snapshot ? "snapshot" : "direct";
    const auto flags  = [&](bool attention_addend) {
        return std::to_string(rr.data != nullptr) + ',' + std::to_string(ca.data != nullptr) + ',' +
               std::to_string(attention_addend);
    };
    if (o.mode == "all" || o.mode == "standalone") {
        row.op = "short_conv_residual";
        benchmark.run(row, conv_bytes, standalone, restore, flags(false));
    }
    if (o.mode == "standalone") return;

    auto stack    = o.stacked ? make_bf16(count * o.stacked, 515) : DeviceBuffer{};
    auto a        = o.attn_add ? make_bf16(count, 516) : DeviceBuffer{};
    auto v        = make_f32(std::size_t(d) * q, 517, -1.0f, 1.0f);
    auto n        = o.norm ? make_bf16(std::size_t(d) * q, 518, 0.5f, 1.5f) : DeviceBuffer{};
    auto y        = make_buffer(count * q * (o.norm ? 2 : 4));
    const auto ss = o.stacked ? Tensor(stack.span(), DType::BF16, {d, m, o.stacked}) : Tensor{};
    const auto aa = o.attn_add ? Tensor(a.span(), DType::BF16, shape) : Tensor{};
    const auto flat_aa = o.attn_add ? aa.reshape({d, m}) : Tensor{};
    const Tensor queries(v.span(), DType::FP32, {d, q});
    const auto norm = o.norm ? Tensor(n.span(), DType::BF16, {d, q}) : Tensor{};
    Tensor out(y.span(), o.norm ? DType::BF16 : DType::FP32, {d, m, q});
    const auto flat_co = co.reshape({d, m});
    const ops::BlockAttnResParameters params{1.0f / std::sqrt(float(d)), 1e-6f, 1e-6f, true};
    const auto fused = [&](CudaExecutionView cuda) {
        if (packed)
            ops::short_conv_block_attn_res(tx, tw, view, source, dest, rr, ca, ss, aa, queries,
                                           norm, params, co, out, cuda);
        else if (o.snapshot)
            ops::short_conv_block_attn_res_snapshot(tx, tw, source, snapshots, {}, slots, rr, ca,
                                                    ss, aa, queries, norm, params, co, out, cuda);
        else
            ops::short_conv_block_attn_res(tx, tw, source, dest, {}, rr, ca, ss, aa, queries, norm,
                                           params, co, out, cuda);
    };
    const auto unfused = [&](CudaExecutionView cuda) {
        standalone(cuda);
        ops::block_attn_res(ss, flat_co, flat_aa, queries, norm, params, out, cuda);
    };
    const auto mix_bytes       = stack.size() + a.size() + v.size() + n.size() + y.size();
    row.format                 = o.norm ? "bf16" : "fp32";
    row.n                      = o.stacked;
    row.k                      = q;
    const auto attention_flags = flags(o.attn_add);
    if (o.mode == "all" || o.mode == "fused") {
        row.op = "short_conv_block_attn_res";
        benchmark.run(row, conv_bytes + mix_bytes, fused, restore, attention_flags);
    }
    if (o.mode == "all" || o.mode == "unfused") {
        row.op = "short_conv_then_block_attn_res";
        benchmark.run(row, conv_bytes + mix_bytes + conv.size(), unfused, restore, attention_flags);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        Options o;
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--help") {
                std::cout
                    << "short_conv [--d D] [--width W[,W...] --batch B | --ragged PARTITIONS]\n"
                       "  --snapshot 0|1 --state-form "
                    << ragged::state_form_options
                    << " --mode all|standalone|fused|unfused\n"
                       "  --stack N --queries Q --norm 0|1 --conv-addend 0|1 "
                       "--residual 0|1 --attn-addend 0|1\n";
                measurement::help();
                return 0;
            }
            const auto value = measurement::take(argc, argv, i);
            if (flag == "--d")
                o.d = measurement::integer(value);
            else if (flag == "--width")
                o.widths = measurement::integers(value);
            else if (flag == "--batch")
                o.batch = measurement::integer(value);
            else if (flag == "--ragged")
                o.segments = ragged::parse_ragged_specs(value);
            else if (flag == "--state-form")
                o.state_form = ragged::parse_state_form(value);
            else if (flag == "--mode") {
                if (value != "all" && value != "standalone" && value != "fused" &&
                    value != "unfused")
                    throw std::invalid_argument("unknown mode " + std::string(value));
                o.mode = value;
            } else if (flag == "--stack")
                o.stacked = measurement::integer(value, 0);
            else if (flag == "--queries")
                o.q = measurement::integer(value);
            else if (flag == "--snapshot")
                o.snapshot = measurement::integer(value, 0, 1);
            else if (flag == "--norm")
                o.norm = measurement::integer(value, 0, 1);
            else if (flag == "--attn-addend")
                o.attn_add = measurement::integer(value, 0, 1);
            else if (flag == "--conv-addend")
                o.conv_add = measurement::integer(value, 0, 1);
            else if (flag == "--residual")
                o.res = measurement::integer(value, 0, 1);
            else if (!measurement::parse(o.timing, flag, value))
                throw std::invalid_argument("unknown option " + std::string(flag));
        }
        measurement::LogicalBenchmark benchmark(o.timing,
                                                "conv_residual,conv_addend,attnres_addend");
        if (o.segments.empty()) {
            for (int width : o.widths) {
                o.width = width;
                run(o, {}, benchmark);
            }
        } else
            for (const auto& lengths : o.segments) run(o, lengths, benchmark);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
