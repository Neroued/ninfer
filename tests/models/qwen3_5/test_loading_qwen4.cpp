// Qwen4Exp (Qwen3.8-Flash-Next) loading: config interpretation, logical binding, residency of
// the routed experts (host-mapped, stacked NVFP4 banks) and of the n-gram table (file-mapped), and
// the byte spans an expert cache copies, checked against an expectation file written from the
// source checkpoint by qwen4_load_interop.py (an oracle independent of the converter and of this
// loader). --plan-only binds without a device and prints the byte inventory by residency class;
// that mode also serves the real 125B artifacts.

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "models/qwen3_5/load.h"

#include <cuda_runtime.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;
using ninfer::test::artifact_fixture::Json;
using ninfer::test::artifact_fixture::require;

void require_equal(std::uint64_t got, std::uint64_t want, const std::string& what) {
    if (got != want) {
        throw std::runtime_error(what + ": got " + std::to_string(got) + ", expected " +
                                 std::to_string(want));
    }
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::byte> out(data.size());
    std::memcpy(out.data(), data.data(), data.size());
    return out;
}

std::vector<std::byte> hex_bytes(const std::string& hex) {
    std::vector<std::byte> out(hex.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = std::byte(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
    }
    return out;
}

const qwen::MoeWeights& moe_of(const qwen::ModelWeights& w, const std::string& component,
                               std::uint32_t layer) {
    if (component == "mtp") { return std::get<qwen::MoeWeights>(w.qwen4_mtp.value().layer.ffn); }
    return std::get<qwen::MoeWeights>(w.qwen4.value().layers.at(layer).ffn);
}

// Config and binding checks that need no device.
void check_plan(const qwen::LoadPlan& plan, const Json& expect, const LoadOptions& options) {
    const auto& text = plan.config().text;
    require(text.qwen4() && text.architecture == Architecture::Qwen4Exp, "not Qwen4Exp");
    require_equal(text.hidden_size, expect.at("hidden"), "hidden_size");
    require_equal(text.num_hidden_layers, expect.at("layers"), "num_hidden_layers");
    require_equal(text.vocab_size, expect.at("vocab"), "vocab_size");
    const auto& moe = std::get<qwen::MoeConfig>(text.ffn);
    require_equal(moe.num_experts, expect.at("experts"), "num_experts");
    require_equal(moe.num_experts_per_tok, expect.at("experts_per_token"), "num_experts_per_tok");
    require_equal(moe.moe_intermediate_size, expect.at("moe_intermediate"), "moe width");
    require_equal(moe.shared_expert_intermediate_size, expect.at("shared_intermediate"),
                  "shared expert width");
    require_equal(text.hyper_connection->hc_count, expect.at("hc_count"), "hc_count");
    require_equal(text.hyper_connection->hc_lowrank, expect.at("hc_lowrank"), "hc_lowrank");
    require_equal(text.residual_width(), std::uint64_t(expect.at("hc_count")) * text.hidden_size,
                  "residual width");
    require(text.gdn_gate == qwen::GateActivation::Sigmoid, "GDN output gate is not sigmoid");
    const auto& qsa = expect.at("qsa");
    require_equal(text.sparse_attention->indexer_heads, qsa.at("heads"), "indexer heads");
    require_equal(text.sparse_attention->indexer_head_dim, qsa.at("head_dim"), "indexer dim");
    require_equal(text.sparse_attention->budget, qsa.at("budget"), "indexer budget");
    require_equal(text.sparse_attention->compress_ratio, qsa.at("compress_ratio"),
                  "indexer compress ratio");
    std::uint32_t full = 0;
    for (std::size_t i = 0; i < text.layer_types.size(); ++i) {
        const bool attention = text.layer_types[i] == qwen::MixerKind::FullAttention;
        require(attention == (expect.at("layer_types")[i] == "full_attention"), "layer types");
        full += attention;
    }
    require_equal(text.full_attention_layers, full, "QSA layer count");
    // PLE hash constants: derived by the loader from seed/base/divisor, compared with values the
    // checkpoint states (or an independent implementation for the synthetic one).
    const auto& ple  = text.ple.value();
    const auto& want = expect.at("ple");
    require_equal(ple.layer, want.at("layer"), "PLE block");
    require_equal(ple.table_rows, want.at("rows"), "PLE table rows");
    require_equal(ple.row_width(), want.at("row_width"), "PLE row width");
    require(ple.multipliers == want.at("multipliers").get<std::vector<std::uint64_t>>(),
            "PLE hash multipliers differ");
    require(ple.head_moduli == want.at("moduli").get<std::vector<std::uint64_t>>(),
            "PLE head moduli differ");
    require(ple.head_offsets == want.at("offsets").get<std::vector<std::uint64_t>>(),
            "PLE head offsets differ");

    const auto& w = plan.weights();
    require(w.qwen4 && w.qwen4->layers.size() == text.num_hidden_layers, "Qwen4 blocks unbound");
    require(w.qwen4_mtp.has_value() == options.mtp(), "MTP layer selection differs");
    require(!w.mtp && w.text.layers.empty(), "Qwen3.5 block bindings used for Qwen4Exp");
    std::size_t ple_blocks = 0;
    for (std::size_t i = 0; i < w.qwen4->layers.size(); ++i) {
        const auto& block = w.qwen4->layers[i];
        require(std::holds_alternative<qwen::QsaWeights>(block.mixer) ==
                    (text.layer_types[i] == qwen::MixerKind::FullAttention),
                "mixer kind differs from layer_types");
        if (block.ple) {
            ++ple_blocks;
            require(i == ple.layer, "PLE bound on the wrong block");
            const auto& table = plan.parameter(block.ple->table);
            require(table.residency == artifact::Residency::HostFile, "table not file-mapped");
        }
    }
    require_equal(ple_blocks, 1, "PLE blocks");
    for (const auto& [name, format] : expect.at("formats").items()) {
        const auto& binding  = plan.materialization().source->directory().bindings.at(name);
        const auto& geometry = plan.materialization().source->geometry(binding.parts.at(0).object);
        require(artifact::format_name(geometry.format) == format.get<std::string>(),
                ("format of " + name).c_str());
    }
}

struct Inventory {
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> by_class; // objects, bytes
};

// Residency of every planned parent: device (per format), host-mapped, file-mapped, host.
Inventory inventory(const qwen::LoadPlan& plan) {
    Inventory out;
    const auto& reader = *plan.materialization().source;
    const auto add     = [&](const std::string& key, std::uint64_t bytes) {
        auto& entry = out.by_class[key];
        ++entry.first;
        entry.second += bytes;
    };
    for (const auto& item : plan.materialization().device_objects) {
        add("device " + std::string(artifact::format_name(reader.geometry(item.object).format)),
            item.bytes);
    }
    for (const auto& item : plan.materialization().host_mapped_objects) {
        add("host-mapped " +
                std::string(artifact::format_name(reader.geometry(item.object).format)),
            item.bytes);
    }
    for (const auto handle : plan.materialization().file_objects) {
        add("file-mapped", artifact::object_bytes(reader.directory().object(handle)));
    }
    for (const auto& item : plan.materialization().host_objects) {
        add("host (resources, values)",
            artifact::object_bytes(reader.directory().object(item.object)));
    }
    return out;
}

void print_inventory(const Inventory& inventory) {
    std::uint64_t total = 0;
    for (const auto& [key, entry] : inventory.by_class) {
        std::cout << "  " << key << ": " << entry.first << " objects, " << entry.second << " B ("
                  << double(entry.second) / double(1ULL << 30) << " GiB)\n";
        total += entry.second;
    }
    std::cout << "  total planned: " << total << " B\n";
}

void check_residency(const qwen::LoadPlan& plan, const Json& expect) {
    const auto& reader = *plan.materialization().source;
    std::set<std::size_t> mapped;
    for (const auto& item : plan.materialization().host_mapped_objects) {
        mapped.insert(item.object.index);
    }
    std::set<std::size_t> device;
    for (const auto& item : plan.materialization().device_objects) {
        device.insert(item.object.index);
    }
    const auto& w      = plan.weights();
    const auto experts = std::get<qwen::MoeConfig>(plan.config().text.ffn).num_experts;
    std::set<std::size_t> banks;
    const auto check_moe = [&](const qwen::MoeWeights& moe) {
        require_equal(moe.experts.size(), experts, "routed experts bound");
        for (const auto& expert : moe.experts) {
            for (const auto id : {expert.gate, expert.up, expert.down}) {
                const auto& reference = plan.parameter(id);
                require(reference.binding.parts.size() == 1, "expert split over parents");
                const auto object = reference.binding.parts[0].object.index;
                require(mapped.contains(object) && !device.contains(object),
                        "routed expert bank is not host-mapped");
                banks.insert(object);
            }
        }
        for (const auto id : {moe.router, moe.shared_score, moe.shared.gate, moe.shared.down}) {
            const auto object = plan.parameter(id).binding.parts.at(0).object.index;
            require(device.contains(object), "dense MoE weight is not on the device");
        }
    };
    for (const auto& block : w.qwen4->layers) { check_moe(std::get<qwen::MoeWeights>(block.ffn)); }
    if (w.qwen4_mtp) { check_moe(std::get<qwen::MoeWeights>(w.qwen4_mtp->layer.ffn)); }
    // Two banks per routed layer, each a stacked plane with one divisor per source matrix.
    const auto layers = w.qwen4->layers.size() + (w.qwen4_mtp ? 1 : 0);
    require_equal(banks.size(), 2 * layers, "expert banks");
    for (const auto index : banks) {
        const auto& g = reader.geometry(artifact::ObjectHandle{index});
        require(g.format == QType::NVFP4, "expert bank is not NVFP4");
        // One divisor per matrix (this tree's recipes), or one per bank (ninfer-ext's MTP bank).
        const auto per_matrix = g.shape[0] / g.rows_per_divisor() == g.divisor_count &&
                                (g.divisor_count == 2 * experts || g.divisor_count == experts);
        require(per_matrix || (g.divisor_count == 1 && !expect.value("per_matrix_divisors", true)),
                "expert bank does not keep one divisor per matrix");
    }
    const auto embedding = plan.parameter(w.text.token_embedding).binding.parts[0].object.index;
    require(mapped.contains(embedding) == expect.value("embedding_host", false),
            "token embedding residency differs from --embedding-host");
}

void check_materialized(const qwen::Model& model, const Json& expect,
                        const std::filesystem::path& directory) {
    const auto& w = model.weights();
    for (const auto& slot : expect.at("expert_slots")) {
        const auto& moe = moe_of(w, slot.at("component").get<std::string>(),
                                 slot.at("layer").get<std::uint32_t>());
        const auto spans =
            qwen::routed_expert_spans(model, moe, slot.at("expert").get<std::uint32_t>());
        std::vector<std::byte> got;
        for (const auto* bank : {&spans.gate_up, &spans.down}) {
            for (const auto& span : bank->spans) {
                // Host-mapped: the device pointer is the host pointer under UVA; read it on host.
                const auto* data = bank->parent->data + span.offset;
                got.insert(got.end(), data, data + span.bytes);
            }
        }
        const auto want = read_file(directory / slot.at("file").get<std::string>());
        require_equal(spans.bytes(), want.size(), "expert slot bytes");
        require(got == want, ("expert slot bytes differ for " + slot.dump()).c_str());
        require_equal(spans.gate_up.spans.size() + spans.down.spans.size(), 6,
                      "spans per expert (codes, scales, divisors per bank)");
    }
    const auto& block  = w.qwen4->layers.at(model.config().text.ple->layer);
    const auto& table  = model.file_weight(block.ple->table);
    const auto& g      = table.mapped_geometry;
    const auto width   = g.shape.at(1);
    const auto read_at = [&](std::uint64_t offset, std::uint64_t bytes) {
        std::vector<std::byte> out;
        for (const auto& segment : table.mapped) {
            const auto begin = std::max(offset, segment.object_offset);
            const auto end   = std::min(offset + bytes, segment.object_offset + segment.bytes);
            if (begin < end) {
                const auto* data = segment.data + (begin - segment.object_offset);
                out.insert(out.end(), data, data + (end - begin));
            }
        }
        return out;
    };
    for (const auto& row : expect.at("table_rows")) {
        const auto index = row.at("row").get<std::uint64_t>();
        require(read_at(index * width, width) == hex_bytes(row.at("codes")),
                "mapped n-gram codes differ");
        require(read_at(g.scale_offset + 2 * index, 2) == hex_bytes(row.at("scale")),
                "mapped n-gram row scale differs");
    }
    require(model.storage_stats().mapped_file_bytes == g.bytes, "mapped bytes differ");
    require(model.storage_stats().host_mapped_bytes > 0, "no host-mapped backing");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path, expectations;
        LoadOptions options;
        bool plan_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--expect") {
                expectations = value();
            } else if (arg == "--mtp") {
                options.speculative = SpeculativeBackend::Mtp;
            } else if (arg == "--embedding-host") {
                options.embedding_host = true;
            } else if (arg == "--plan-only") {
                plan_only = true;
            } else {
                throw std::invalid_argument("usage: --artifact PATH [--expect JSON] [--mtp] "
                                            "[--embedding-host] [--plan-only]");
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply --artifact (qwen4_load_interop.py drives this test)\n";
            return 77;
        }
        Json expect;
        if (!expectations.empty()) {
            std::ifstream in(expectations);
            expect                   = Json::parse(in);
            expect["embedding_host"] = options.embedding_host;
        }
        artifact::Reader reader(path);
        auto plan = qwen::plan_load(reader, options);
        if (!expect.is_null()) {
            check_plan(plan, expect, options);
            check_residency(plan, expect);
        }
        std::cout << path.filename().string() << ": plan passed, " << plan.parameter_count()
                  << " parameters\n";
        print_inventory(inventory(plan));
        if (plan_only) { return 0; }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "SKIP: materialization needs a CUDA device\n";
            return 77;
        }
        DeviceContext device;
        auto model = qwen::materialize_model(std::move(plan), device);
        if (!expect.is_null()) { check_materialized(*model, expect, expectations.parent_path()); }
        std::cout << "materialized: host-mapped " << model->storage_stats().host_mapped_bytes
                  << " B, file-mapped " << model->storage_stats().mapped_file_bytes << " B, device "
                  << model->storage_stats().device_capacity_bytes << " B\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
