#include "native_artifact.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"

#include <cstring>

namespace ninfer::test::native_moe {
namespace {
namespace native = ninfer::test::native_moe;
namespace qw     = ninfer::test::quantized_weight;

qw::PackedWeightBank read_bank(artifact::Reader& reader, const std::string& prefix, bool main,
                               bool ep, int rank, bool gate) {
    const int intermediate = ep ? 2048 : 1024;
    const int n = gate ? 2 * intermediate : 4096, k = gate ? 4096 : intermediate;
    std::vector<std::uint32_t> ids;
    for (int e = 0; e < 288; ++e)
        if (!ep || e % 2 == rank) ids.push_back(e);
    if (ep) std::reverse(ids.begin(), ids.end());
    const auto binding = [&](const char* name) -> const artifact::Binding& {
        const auto& b = reader.directory().bindings.at(prefix + "/experts/" + name);
        if (b.parts.size() != 288) throw std::invalid_argument("expected 288 native expert parts");
        return b;
    };
    const auto& first  = binding(gate ? "gate" : "down");
    const auto& second = gate ? binding("up") : first;
    std::vector<artifact::MatrixSlice> slices;
    for (std::size_t slot = 0; slot < ids.size(); ++slot) {
        for (int part = 0; part < (gate ? 2 : 1); ++part) {
            const auto& p       = (part ? second : first).parts[ids[slot]];
            const auto& g       = reader.geometry(p.object);
            const auto source_k = g.shape.back();
            if (g.format != QType::NVFP4_MUL || p.begin % source_k ||
                p.end - p.begin != 4096ULL * 2048 ||
                g.layout != (main ? QuantLayout::Csf32 : QuantLayout::BatchedBlockScaleK16M128x4))
                throw std::invalid_argument("artifact expert shape or representation differs");
            const auto row    = p.begin / source_k + (gate && !ep ? rank * intermediate : 0);
            const auto column = !gate && !ep ? rank * intermediate : 0;
            slices.push_back({p.object, row, std::uint64_t(column),
                              std::uint64_t(gate ? intermediate : n), std::uint64_t(k),
                              slot * n + part * intermediate, 0});
        }
    }
    const std::array<std::uint64_t, 3> shape{ids.size(), std::uint64_t(n), std::uint64_t(k)};
    const std::array requests{
        artifact::TensorRequest{artifact::slice_tensor(reader, shape, slices), Device::cpu()}};
    auto plan               = artifact::plan_materialization(reader, requests);
    const auto storage      = plan.requests[0];
    const auto materialized = artifact::materialize(reader, std::move(plan));
    const auto& parent      = materialized.parent(storage, Device::cpu());
    qw::PackedWeightBank bank;
    bank.geometry = parent.geometry;
    bank.payload.resize(parent.geometry.bytes);
    std::memcpy(bank.payload.data(), parent.data, bank.payload.size());
    bank.matrix_scales.assign(parent.matrix_scales.begin(), parent.matrix_scales.end());
    bank.expert_indices.assign(288, UINT32_MAX);
    for (std::size_t slot = 0; slot < ids.size(); ++slot) bank.expert_indices[ids[slot]] = slot;
    if (main) {
        artifact::Binder binder(reader);
        const auto parameter = prefix + (gate ? "/experts/gate" : "/experts/down");
        const auto input =
            gate ? prefix.substr(0, prefix.size() - 4) + "/moe_input" : prefix + "/experts/product";
        const auto values = binder.values(
            binder.use(parameter, input).auxiliaries.at("activation_input_multiplier"),
            QType::FP32);
        if (values.elements != 288) throw std::invalid_argument("expected per-expert calibration");
        bank.activation_scales.resize(288);
        std::memcpy(bank.activation_scales.data(), values.data.data(), values.data.size());
    }
    const auto& g = bank.geometry;
    for (std::size_t slot = 0; slot < ids.size(); ++slot) {
        std::uint64_t bytes = g.code_bytes / ids.size() + 4;
        if (main) {
            const auto first_page = slot * g.csf.pages_per_matrix;
            const auto last_page  = first_page + g.csf.pages_per_matrix - 1;
            const auto begin =
                qw::detail::load_u32_le(bank.payload, g.csf.page_bases_offset + first_page * 4);
            const auto end =
                qw::detail::load_u32_le(bank.payload, g.csf.page_bases_offset + last_page * 4) +
                2 * qw::detail::load_u16_le(bank.payload,
                                            g.csf.offsets_offset +
                                                ((slot + 1) * g.csf.offsets_per_matrix - 1) * 2);
            bytes += n + g.csf.nibble_bytes_per_matrix + g.csf.offsets_per_matrix * 2 +
                     g.csf.pages_per_matrix * 4 + end - begin;
        } else {
            bytes += g.scale_bytes / ids.size();
        }
        bank.matrix_bytes.push_back(bytes);
    }
    return bank;
}
} // namespace

WeightBanks read_artifact_weights(const std::filesystem::path& path, Profile profile, int rank,
                                  int layer) {
    const bool main   = profile == native::Profile::MainTp || profile == native::Profile::MainEp;
    const bool ep     = profile == native::Profile::MainEp || profile == native::Profile::MtpEp;
    const auto prefix = main ? "text/layers/" + std::to_string(layer) + "/moe" : "mtp/layers/0/moe";
    artifact::Reader reader(path);
    return {read_bank(reader, prefix, main, ep, rank, true),
            read_bank(reader, prefix, main, ep, rank, false)};
}
} // namespace ninfer::test::native_moe
