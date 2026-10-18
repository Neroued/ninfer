#include "models/qwen3_5/program/text_graph.h"

#include "core/decode_graph.h"
#include "models/qwen3_5/program/graphs.h"

#include <algorithm>
#include <stdexcept>
#include <variant>

namespace ninfer::models::qwen3_5::execution {
namespace {

struct Segment {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;

    void capture(CudaExecutionView cuda, const std::function<void()>& body) {
        definition.capture(cuda.stream(), body);
        executable.instantiate(definition);
        executable.upload(cuda.stream());
    }
};

struct Resident {
    Segment main;
    std::vector<DecodeGraphChild> mixers;
    std::shared_ptr<std::vector<std::vector<DecodeGraphDefinition>>> profiles;
};

struct LayerDefinition {
    DecodeGraphDefinition resident;
    std::shared_ptr<const moe::LayerGraphDefinition> offloaded;
};

struct LayerExecutable {
    DecodeGraphExecutable resident;
    std::unique_ptr<moe::LayerGraphExecutable> offloaded;
};

struct OffloadedTarget {
    std::vector<std::vector<LayerDefinition>> profiles;
    std::vector<LayerExecutable> layers;
    std::size_t installed = 0;
};

struct Offloaded {
    Segment prefix, suffix;
    std::shared_ptr<OffloadedTarget> target;
    Tensor residual;
    moe::Schedule* schedule = nullptr;
    std::optional<moe::Schedule::Activity> activity;
    std::optional<CudaExecutionView> cuda;
    std::size_t next_layer = 0;
    bool waiting           = false;

    bool advance() {
        if (!cuda) throw std::logic_error("text graph is not active");
        while (next_layer < target->layers.size()) {
            auto& layer = target->layers[next_layer];
            if (layer.offloaded) {
                const bool complete =
                    waiting ? layer.offloaded->poll() : layer.offloaded->start(residual, *cuda);
                if (!complete) {
                    waiting = true;
                    return false;
                }
                waiting = false;
            } else {
                layer.resident.launch(cuda->stream());
            }
            ++next_layer;
        }
        suffix.executable.launch(cuda->stream());
        activity.reset();
        cuda.reset();
        return true;
    }

    void cancel() {
        if (waiting) target->layers[next_layer].offloaded->cancel();
        waiting = false;
        activity.reset();
        cuda.reset();
    }
};

ops::CausalAttentionExecutionEnvelope envelope(const GraphProfile& profile, std::uint32_t width,
                                               std::uint32_t capacity) {
    return {1, static_cast<std::uint32_t>(
                   std::min<std::uint64_t>(capacity, std::uint64_t(profile.max) + width))};
}

} // namespace

struct TextGraph::Impl {
    std::vector<GraphProfile> profiles;
    std::variant<Resident, Offloaded> graph;
    std::size_t installed      = 0;
    std::size_t workspace_peak = 0;
    std::uint32_t width        = 0;
};

TextGraph::TextGraph()                                = default;
TextGraph::~TextGraph()                               = default;
TextGraph::TextGraph(TextGraph&&) noexcept            = default;
TextGraph& TextGraph::operator=(TextGraph&&) noexcept = default;

void TextGraph::prepare(TextRun& run, const Parameters& parameters, moe::Schedule* schedule,
                        std::span<const GraphProfile> profiles, std::uint32_t width,
                        const std::function<void()>& prologue,
                        const std::function<void()>& epilogue, MemoryArena& workspace,
                        DeviceContext& device, const TextGraph* shared) {
    if (impl_) throw std::logic_error("text graph is already prepared");
    if (profiles.empty() || profiles.front().min != 0 || width == 0) {
        throw std::invalid_argument("text graph profiles are invalid");
    }
    auto result = std::make_unique<Impl>();
    result->profiles.assign(profiles.begin(), profiles.end());
    result->width = width;
    if (shared && (!shared->impl_ || shared->impl_->profiles != result->profiles ||
                   shared->impl_->width != width)) {
        throw std::invalid_argument("shared text graphs must have identical execution geometry");
    }
    const auto capacity = profiles.back().max + 1;
    const auto cuda     = device.execution_view();
    const auto layers   = parameters.text.layers.size();
    run.set_attention_envelope(envelope(profiles.front(), width, capacity));
    prologue();
    bool ready = run.start();
    while (!ready) {
        if (schedule) schedule->synchronize();
        device.synchronize();
        ready = run.poll();
    }
    epilogue();
    device.synchronize();

    if (!schedule) {
        auto& graph    = result->graph.emplace<Resident>();
        graph.profiles = shared ? std::get<Resident>(shared->impl_->graph).profiles
                                : std::make_shared<std::vector<std::vector<DecodeGraphDefinition>>>(
                                      profiles.size());
        std::vector<std::uint32_t> attention_layers;
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            if (std::holds_alternative<AttentionParameters>(parameters.text.layers[layer].mixer)) {
                attention_layers.push_back(layer);
            }
        }
        for (std::size_t profile = 0; !shared && profile < profiles.size(); ++profile) {
            run.set_attention_envelope(envelope(profiles[profile], width, capacity));
            auto& definitions = (*graph.profiles)[profile];
            definitions.resize(attention_layers.size());
            for (std::size_t index = 0; index < attention_layers.size(); ++index) {
                definitions[index].capture(cuda.stream(),
                                           [&] { run.mixer(attention_layers[index]); });
            }
        }
        graph.main.capture(cuda, [&] {
            prologue();
            run.prefix();
            std::size_t attention = 0;
            for (std::uint32_t layer = 0; layer < layers; ++layer) {
                if (attention < attention_layers.size() && attention_layers[attention] == layer) {
                    graph.mixers.push_back(
                        graph.profiles->front()[attention++].append_to(cuda.stream()));
                } else {
                    run.mixer(layer);
                }
                run.post_mixer(layer);
                run.capture_features(layer);
            }
            run.finish();
            epilogue();
        });
        for (std::size_t profile = 1; profile < graph.profiles->size(); ++profile) {
            const auto& definitions = (*graph.profiles)[profile];
            for (std::size_t index = 0; index < definitions.size(); ++index) {
                graph.main.executable.update_child(graph.mixers[index], definitions[index]);
            }
            graph.main.executable.upload(cuda.stream());
            device.synchronize();
        }
        if (graph.profiles->size() > 1) {
            for (std::size_t index = 0; index < graph.mixers.size(); ++index)
                graph.main.executable.update_child(graph.mixers[index],
                                                   graph.profiles->front()[index]);
            graph.main.executable.upload(cuda.stream());
        }
    } else {
        auto& graph    = result->graph.emplace<Offloaded>();
        graph.residual = run.residual();
        graph.schedule = schedule;
        graph.target   = shared ? std::get<Offloaded>(shared->impl_->graph).target
                                : std::make_shared<OffloadedTarget>();
        graph.prefix.capture(cuda, [&] {
            prologue();
            run.prefix();
        });
        auto& target = *graph.target;
        if (!shared) target.profiles.resize(profiles.size());
        for (std::size_t profile = 0; !shared && profile < profiles.size(); ++profile) {
            run.set_attention_envelope(envelope(profiles[profile], width, capacity));
            auto& definitions = target.profiles[profile];
            definitions.resize(layers);
            for (std::uint32_t layer = 0; layer < layers; ++layer) {
                auto& definition = definitions[layer];
                const auto* routed =
                    std::get_if<moe::Parameters>(&parameters.text.layers[layer].ffn);
                if (routed) {
                    const auto prior = profile ? target.profiles.front()[layer].offloaded : nullptr;
                    definition.offloaded = schedule->capture_layer(
                        routed->bank, cuda, run.residual().ne[1], [&] { run.mixer(layer); },
                        [&](const moe::GpuCall& call) { run.post_mixer(layer, &call); },
                        [&] { run.capture_features(layer); }, prior);
                } else {
                    definition.resident.capture(cuda.stream(), [&] {
                        run.mixer(layer);
                        run.post_mixer(layer);
                        run.capture_features(layer);
                    });
                }
            }
        }
        graph.suffix.capture(cuda, [&] {
            run.finish();
            epilogue();
        });
        if (!shared) target.layers.resize(layers);
        for (std::size_t layer = 0; !shared && layer < layers; ++layer) {
            auto& executable       = target.layers[layer];
            const auto& definition = target.profiles.front()[layer];
            if (definition.offloaded)
                executable.offloaded = definition.offloaded->instantiate();
            else
                executable.resident.instantiate(definition.resident);
        }
        for (std::size_t profile = 0; !shared && profile < profiles.size(); ++profile) {
            const auto& definitions = target.profiles[profile];
            for (std::size_t layer = 0; layer < layers; ++layer) {
                auto& executable = target.layers[layer];
                if (executable.offloaded) {
                    executable.offloaded->update(*definitions[layer].offloaded);
                    executable.offloaded->upload(cuda.stream());
                } else {
                    executable.resident.update(definitions[layer].resident);
                    executable.resident.upload(cuda.stream());
                }
            }
            device.synchronize();
        }
        for (std::size_t layer = 0; !shared && layer < layers; ++layer) {
            auto& executable       = target.layers[layer];
            const auto& definition = target.profiles.front()[layer];
            if (executable.offloaded)
                executable.offloaded->update(*definition.offloaded);
            else
                executable.resident.update(definition.resident);
        }
    }
    device.synchronize();
    result->workspace_peak = workspace.peak_used();
    impl_                  = std::move(result);
}

void TextGraph::select(std::uint32_t frontier) {
    if (!impl_) throw std::logic_error("text graph has not been prepared");
    const auto found =
        std::find_if(impl_->profiles.begin(), impl_->profiles.end(), [&](const auto& profile) {
            return profile.min <= frontier && frontier <= profile.max;
        });
    if (found == impl_->profiles.end())
        throw std::out_of_range("text graph frontier exceeds capacity");
    const auto index = static_cast<std::size_t>(found - impl_->profiles.begin());
    if (auto* graph = std::get_if<Resident>(&impl_->graph)) {
        if (index == impl_->installed) return;
        for (std::size_t layer = 0; layer < graph->mixers.size(); ++layer) {
            graph->main.executable.update_child(graph->mixers[layer],
                                                (*graph->profiles)[index][layer]);
        }
    } else {
        auto& offload = std::get<Offloaded>(impl_->graph);
        if (offload.cuda) throw std::logic_error("cannot update an active text graph");
        auto& target = *offload.target;
        if (target.installed != index) {
            for (std::size_t layer = 0; layer < target.layers.size(); ++layer) {
                auto& executable       = target.layers[layer];
                const auto& definition = target.profiles[index][layer];
                if (executable.offloaded)
                    executable.offloaded->update(*definition.offloaded);
                else
                    executable.resident.update(definition.resident);
            }
            target.installed = index;
        }
    }
    impl_->installed = index;
}

bool TextGraph::start(CudaExecutionView cuda) {
    if (!impl_) throw std::logic_error("text graph has not been prepared");
    if (auto* graph = std::get_if<Resident>(&impl_->graph)) {
        graph->main.executable.launch(cuda.stream());
        return true;
    }
    auto& graph = std::get<Offloaded>(impl_->graph);
    if (graph.cuda) throw std::logic_error("text graph is already active");
    graph.activity.emplace(std::move(*graph.schedule->begin(graph.residual.ne[1])));
    graph.cuda       = cuda;
    graph.next_layer = 0;
    graph.waiting    = false;
    graph.prefix.executable.launch(cuda.stream());
    return graph.advance();
}

bool TextGraph::poll() {
    if (!impl_) throw std::logic_error("text graph has not been prepared");
    if (auto* graph = std::get_if<Offloaded>(&impl_->graph)) {
        if (graph->cuda) return graph->advance();
    }
    return true;
}

void TextGraph::cancel() {
    if (!impl_) return;
    if (auto* graph = std::get_if<Offloaded>(&impl_->graph)) graph->cancel();
}

std::size_t TextGraph::workspace_peak() const noexcept { return impl_ ? impl_->workspace_peak : 0; }

} // namespace ninfer::models::qwen3_5::execution
