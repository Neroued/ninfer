#include "models/glm5next/load.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

namespace {
using namespace ninfer;
using namespace ninfer::models;
namespace glm = models::glm5next;

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<std::byte> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file.good(), "missing oracle: " + path.string());
    std::vector<std::byte> result(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()), result.size());
    require(file.good(), "cannot read oracle: " + path.string());
    return result;
}

LoadOptions options(bool mtp, bool vision, bool optimized = false) {
    return {.vision        = vision,
            .speculative   = mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
            .proposal_head = optimized ? ProposalHead::Optimized : ProposalHead::Full};
}

DeploymentOptions deployment() {
    DeploymentOptions d;
    d.control_host = "127.0.0.1";
    d.parallel     = {.tp = 2, .ep = 2, .dp = 1, .pp = 1, .adp = 2};
    d.nodes        = {{0, {}, "/worker", "/model.ninfer", {}},
                      {1, "rank1", "/worker", "/model.ninfer", {}}};
    d.ranks        = {{0, 0, 0}, {1, 1, 0}};
    return d;
}

void contiguous(std::span<const BoundWeight> weights, std::initializer_list<WeightId> ids) {
    const WeightParent* parent = nullptr;
    std::uint64_t end          = 0;
    for (const auto id : ids) {
        const auto region = contiguous_weight_region(weights[id.index].view);
        require(!parent || (parent == region.parent && region.begin == end),
                "merged projection has a gap or a different parent");
        parent = region.parent;
        end    = region.end;
    }
}

void layouts(std::span<const BoundWeight> data, const glm::ModelWeights& w) {
    const auto& k = std::get<glm::KdaWeights>(w.text.layers[0].mixer);
    contiguous(data, {k.query, k.key, k.value});
    contiguous(data, {k.beta, k.forget_a, k.gate_a});
    const auto mla = [&](const glm::MlaWeights& m) {
        contiguous(data, {m.latent, m.index_key, m.pool_gate, m.query_a, m.index_weight});
        contiguous(data, {m.query_b, m.index_query});
    };
    mla(std::get<glm::MlaWeights>(w.text.layers[1].mixer));
    if (w.mtp) mla(w.mtp->attention);
}

void source_words(const artifact::Reader& reader, const glm::ModelDefinition& definition,
                  const std::filesystem::path& directory) {
    const auto& w = definition.weights();
    const auto& k = std::get<glm::KdaWeights>(w.text.layers[0].mixer);
    const auto& m = std::get<glm::MlaWeights>(w.text.layers[1].mixer);
    for (const auto& [id, name] :
         std::initializer_list<std::pair<WeightId, const char*>>{{w.text.embedding, "embedding"},
                                                                 {w.text.head, "head"},
                                                                 {k.convolution, "convolution"},
                                                                 {m.latent, "latent"},
                                                                 {m.index_key, "index_key"},
                                                                 {m.pool_gate, "pool_gate"},
                                                                 {m.query_a, "query_a"},
                                                                 {m.index_weight, "index_weight"},
                                                                 {m.query_b, "query_b"},
                                                                 {m.index_query, "index_query"},
                                                                 {m.key_up, "key_up"},
                                                                 {m.value_up, "value_up"}}) {
        const auto& parameter = definition.parameter(id);
        require(parameter.binding.parts.size() == 1, "direct source binding must be contiguous");
        const auto& part   = parameter.binding.parts.front();
        const auto& g      = reader.geometry(part.object);
        const auto word    = g.format == QType::FP32 ? 4U : 2U;
        const auto& object = reader.directory().tensor(part.object);
        const auto actual =
            reader.read_range(object.offset + part.begin * word, (part.end - part.begin) * word);
        require(actual == read_bytes(directory / (std::string(name) + ".bin")),
                std::string(name) + " differs from checkpoint layout");
    }
}

void selection_and_planning(const std::filesystem::path& directory) {
    std::shared_ptr<const glm::ModelDefinition> full;
    {
        artifact::Reader reader(directory / "model.ninfer");
        const auto text = glm::read_definition(reader);
        require(!text->weights().mtp && !text->weights().vision && text->expert_banks().size() == 1,
                "text selection retained optional components");
        const auto mtp = glm::read_definition(reader, options(true, false));
        require(mtp->weights().mtp && !mtp->weights().vision, "MTP selection differs");
        full                 = glm::read_definition(reader, options(true, true));
        const auto optimized = glm::read_definition(reader, options(true, true, true));
        const auto& w        = full->weights();
        require(full->config().text.hidden == 128 && full->config().text.layers.size() == 2 &&
                    full->config().text.kda_layers == 1 && full->config().text.mla_layers == 1 &&
                    w.vision && w.mtp && full->expert_banks().size() == 2,
                "GLM components or mathematical configuration differ");
        require(full->input(w.text.head).weight.parts.front().parent->geometry.format ==
                        QType::BF16 &&
                    optimized->input(optimized->weights().mtp->head)
                            .weight.parts.front()
                            .parent->geometry.format == QType::Q4_G64_FP16,
                "Main/full-vocabulary proposal formats differ");
        const auto& k = std::get<glm::KdaWeights>(w.text.layers[0].mixer);
        const auto& d = std::get<glm::DenseWeights>(w.text.layers[0].ffn);
        const auto& m = std::get<glm::MoeWeights>(w.text.layers[1].ffn);
        require(full->input(k.query).weight.parts.front().parent->geometry.format ==
                        QType::Q8_G32_FP16 &&
                    full->input(d.gate).weight.parts.front().parent->geometry.format ==
                        QType::NVFP4 &&
                    full->input(m.shared.gate).weight.parts.front().parent->geometry.format ==
                        QType::Q8_G32_FP16,
                "official recipe formats differ");
        require(full->uses(m.gate).front().policy == ops::LinearPolicy::AllowA4 &&
                    full->uses(m.gate).front().activation_input_divisor == 0.5F &&
                    full->uses(w.mtp->moe.gate).front().policy == ops::LinearPolicy::AllowA8,
                "expert use policy or activation divisor lost");
        source_words(reader, *full, directory);
        layouts(full->weight_data(), w);
        const auto parallel = glm::plan_parallel(*full, deployment());
        require(parallel.ranks.size() == 2 && parallel.topology.ranks[1].adp.rank == 1,
                "ADP topology differs");
        for (const auto& rank : parallel.ranks) {
            auto load       = glm::resolve_rank_load(reader, full, rank, Device::cuda(0));
            const auto data = load.weight_data();
            require(rank.kda_heads.begin == rank.rank && rank.kda_heads.size == 1 &&
                        rank.experts.size() == 2 && rank.experts[0] == std::vector{rank.rank} &&
                        rank.experts[1] == std::vector{rank.rank},
                    "TP/EP ownership differs");
            require(data[k.query.index].view.shape == artifact::Shape{64, 128} &&
                        data[k.convolution.index].view.shape == artifact::Shape{4, 192} &&
                        data[d.down.index].view.shape == artifact::Shape{128, 128} &&
                        data[w.text.head.index].view.shape == artifact::Shape{8, 128},
                    "rank-local weight shapes differ");
            layouts(data, w);
            require(data[w.text.embedding.index].view.parts[0].parent ==
                            data[w.mtp->embedding.index].view.parts[0].parent &&
                        data[w.text.head.index].view.parts[0].parent ==
                            data[w.mtp->head.index].view.parts[0].parent,
                    "Main/MTP sharing duplicated planned storage");
            for (const auto& weight : data)
                for (const auto& part : weight.view.parts)
                    require(!part.parent->data, "planning materialized payload storage");
        }
    }
    require(full->resources().text.chat_template_jinja == "{{ messages }}" &&
                full->resources().processor && full->weights().vision->layers.size() == 1,
            "definition did not retain resources after Reader destruction");
}

struct Encoded {
    WeightGeometry geometry;
    std::vector<std::byte> bytes;
    std::uint64_t begin = 0;
};

Encoded source(const artifact::Reader& reader, const glm::ModelDefinition& definition,
               WeightId id) {
    const auto& parts = definition.parameter(id).binding.parts;
    require(parts.size() == 1, "expected one encoded parent");
    return {reader.geometry(parts[0].object), reader.read_object(parts[0].object), parts[0].begin};
}

Encoded download(const BoundWeight& weight) {
    const auto region  = contiguous_weight_region(weight.view);
    const auto& parent = *region.parent;
    Encoded out{parent.geometry, std::vector<std::byte>(parent.geometry.bytes), region.begin};
    CUDA_CHECK(cudaMemcpy(out.bytes.data(), parent.data, out.bytes.size(), cudaMemcpyDeviceToHost));
    return out;
}

std::uint64_t scale_offset(const WeightGeometry& g, std::uint64_t row, std::uint64_t group) {
    if (g.format == QType::NVFP4) {
        const auto groups = g.padded_columns / 16;
        return g.scale_offset + ((row / 128) * (groups / 4) + group / 4) * 512 + (row % 32) * 16 +
               ((row % 128) / 32) * 4 + group % 4;
    }
    return g.scale_offset + row * g.scale_row_bytes + group * 2;
}

void matrix_words(const BoundWeight& weight, const Encoded& expected, std::uint64_t row_start = 0,
                  std::uint64_t column_start = 0) {
    const auto actual  = download(weight);
    const auto& a      = actual.geometry;
    const auto& e      = expected.geometry;
    const auto columns = weight.view.shape.back();
    const auto rows    = weight_element_count(weight.view.shape) / columns;
    const auto arow    = actual.begin / a.shape.back();
    const auto erow    = expected.begin / e.shape.back() + row_start;
    require(a.format == e.format, weight.name + ": format changed");
    const auto equal = [&](std::uint64_t ao, std::uint64_t eo, std::uint64_t bytes) {
        require(ao + bytes <= actual.bytes.size() && eo + bytes <= expected.bytes.size() &&
                    std::memcmp(actual.bytes.data() + ao, expected.bytes.data() + eo, bytes) == 0,
                weight.name + ": encoded slice differs");
    };
    for (std::uint64_t row = 0; row < rows; ++row) {
        if (a.layout == QuantLayout::Contiguous) {
            const auto word = a.format == QType::FP32 ? 4 : 2;
            equal((arow + row) * a.shape.back() * word,
                  ((erow + row) * e.shape.back() + column_start) * word, columns * word);
            continue;
        }
        const auto bits = a.format == QType::Q8_G32_FP16 ? 8U : 4U;
        equal((arow + row) * a.code_bytes_per_row,
              (erow + row) * e.code_bytes_per_row + column_start * bits / 8, columns * bits / 8);
        const auto group = a.format == QType::NVFP4 ? 16U : a.group_size;
        for (std::uint64_t i = 0; i < columns / group; ++i)
            equal(scale_offset(a, arow + row, i),
                  scale_offset(e, erow + row, column_start / group + i),
                  a.format == QType::NVFP4 ? 1 : 2);
    }
    if (a.format == QType::NVFP4) equal(a.divisor_offset, e.divisor_offset, 4);
}

void materialization(const std::filesystem::path& directory, DeviceContext& device,
                     std::uint32_t size, std::uint32_t rank, bool optimized) {
    std::unique_ptr<glm::ModelShard> model;
    std::map<std::size_t, Encoded> expected;

    struct ExpertWords {
        std::uint32_t bank, expert;
        bool down;
        std::vector<std::byte> codes, scales;
        float divisor;
    };

    std::vector<ExpertWords> expert_words;
    {
        artifact::Reader reader(directory / "model.ninfer");
        auto definition = glm::read_definition(reader, options(true, true, optimized));
        auto plan       = size == 1 ? glm::plan_single_rank(*definition)
                                    : glm::plan_parallel(*definition, deployment()).ranks.at(rank);
        for (std::size_t i = 0; i < definition->parameters().size(); ++i)
            if (definition->parameter({i}).binding.parts.size() == 1)
                expected.emplace(i, source(reader, *definition, {i}));
        for (std::uint32_t b = 0; b < definition->expert_banks().size(); ++b) {
            for (const bool down : {false, true}) {
                const auto& description = definition->expert_banks()[b];
                const auto& bank        = down ? description.down : description.gate_up;
                const auto& object      = reader.directory().tensor(bank.object);
                for (const auto e : plan.experts[b]) {
                    const auto codes = reader.read_range(object.offset + bank.layout.codes.offset +
                                                             e * bank.layout.codes.expert_stride,
                                                         bank.layout.codes.expert_bytes);
                    const auto scales =
                        reader.read_range(object.offset + bank.layout.scales.offset +
                                              e * bank.layout.scales.expert_stride,
                                          bank.layout.scales.expert_bytes);
                    expert_words.push_back(
                        {b, e, down, codes, scales, b == 0 ? 1.0F / (e + 1) : 1.0F});
                }
            }
        }
        model = glm::materialize_model(
            glm::resolve_rank_load(reader, definition, plan, device.device()), device);
    }
    const auto& w   = model->definition().weights();
    const auto& k   = std::get<glm::KdaWeights>(w.text.layers[0].mixer);
    const auto& d   = std::get<glm::DenseWeights>(w.text.layers[0].ffn);
    const auto& m   = std::get<glm::MoeWeights>(w.text.layers[1].ffn);
    const auto part = size == 1 ? 0U : rank * 64U;
    for (const auto id : {k.query, k.key, k.value, k.forget_b, k.gate_b, m.shared.gate, m.shared.up,
                          w.mtp->moe.shared.gate, w.mtp->moe.shared.up})
        matrix_words(model->weight(id), expected.at(id.index), part);
    for (const auto id : {k.output, m.shared.down, w.mtp->moe.shared.down})
        matrix_words(model->weight(id), expected.at(id.index), 0, part);
    for (const auto id : {d.gate, d.up})
        matrix_words(model->weight(id), expected.at(id.index), size == 1 ? 0 : rank * 128);
    matrix_words(model->weight(d.down), expected.at(d.down.index), 0, size == 1 ? 0 : rank * 128);
    matrix_words(model->weight(k.beta), expected.at(k.beta.index), size == 1 ? 0 : rank);
    for (const auto id : {k.forget_a, k.gate_a, w.text.embedding})
        matrix_words(model->weight(id), expected.at(id.index));
    for (const auto id : {w.text.head, w.mtp->head})
        matrix_words(model->weight(id), expected.at(id.index), size == 1 ? 0 : rank * 8);
    const auto& mla = std::get<glm::MlaWeights>(w.text.layers[1].mixer);
    for (const auto id : {mla.latent, mla.index_key, mla.pool_gate, mla.query_a, mla.index_weight,
                          mla.query_b, mla.index_query, mla.key_up, mla.value_up, mla.output,
                          w.vision->patch_embedding, w.vision->layers[0].query_bias})
        matrix_words(model->weight(id), expected.at(id.index));
    const auto convolution = download(model->weight(k.convolution));
    const auto original    = read_bytes(directory / "convolution.bin");
    const auto local       = 128 / size;
    for (unsigned tap = 0; tap < 4; ++tap)
        for (unsigned field = 0; field < 3; ++field)
            require(std::memcmp(convolution.bytes.data() + (tap * 3 * local + field * local) * 4,
                                original.data() + (tap * 384 + field * 128 + rank * local) * 4,
                                local * 4) == 0,
                    "KDA convolution tap/QKV rank slice differs");
    require(model->weight(w.text.embedding).view.parts[0].parent ==
                model->weight(w.mtp->embedding).view.parts[0].parent,
            "embedding storage duplicated");
    if (!optimized)
        require(native_weight(model->input(w.text.head).weight).qdata ==
                    native_weight(model->input(w.mtp->head).weight).qdata,
                "full MTP head duplicated Main storage");
    for (const auto& e : expert_words) {
        const auto& bank = model->experts({e.bank}).view();
        require(bank.count == 2, "local EP bank lost the global routing domain");
        const auto& matrix = e.down ? bank.down : bank.gate_up;
        std::array<std::uint64_t, 2> offsets{};
        CUDA_CHECK(
            cudaMemcpy(offsets.data(), matrix.offsets, sizeof(offsets), cudaMemcpyDeviceToHost));
        for (unsigned id = 0; id < 2; ++id)
            require((offsets[id] != kExpertAbsentOffset) == (size == 1 || id == rank),
                    "EP global-ID presence differs");
        std::vector<std::byte> codes(e.codes.size()), scales(e.scales.size());
        CUDA_CHECK(cudaMemcpy(codes.data(), matrix.codes + offsets[e.expert], codes.size(),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scales.data(), matrix.scales + offsets[e.expert], scales.size(),
                              cudaMemcpyDeviceToHost));
        require(codes == e.codes && scales == e.scales, "resident expert encoded planes changed");
        if (e.bank == 0) {
            float divisor = 0;
            CUDA_CHECK(cudaMemcpy(&divisor, matrix.divisors + e.expert, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            require(divisor == e.divisor, "resident expert divisor lost global-ID addressing");
        }
    }
    require(model->storage_stats().host_capacity_bytes == 0 && model->metadata_bytes() > 0 &&
                model->definition().resources().text.chat_template_jinja == "{{ messages }}",
            "materialized ownership or retained resource lifetime differs");
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: loading-test fixture-directory --host|--gpu");
        const std::filesystem::path directory(argv[1]);
        if (std::string_view(argv[2]) == "--host") {
            selection_and_planning(directory);
        } else {
            require(std::string_view(argv[2]) == "--gpu", "unknown loading test mode");
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
            DeviceContext device;
            materialization(directory, device, 1, 0, false);
            materialization(directory, device, 2, 0, true);
            materialization(directory, device, 2, 1, true);
            materialization(directory, device, 2, 1, false);
        }
        std::cout << "PASS GLM conversion, binding and rank loading\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
