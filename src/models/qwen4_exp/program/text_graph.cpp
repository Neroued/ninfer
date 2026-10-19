#include "models/qwen4_exp/program/text_graph.h"
#include "core/nvtx.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

void GraphSegment::prepare(CudaExecutionView cuda, const std::function<void()>& body) {
    definition.capture(cuda.stream(), body);
    executable.instantiate(definition);
    executable.upload(cuda.stream());
}

void TextGraph::prepare(execution::TextRun& run, const execution::Parameters& parameters,
                        moe::Schedule* experts, std::span<const std::uint32_t> bounds,
                        std::array<ple::DeviceEmbedding, 2> ple, execution::TextOutputs output,
                        Tensor logits, std::int32_t columns, MemoryArena& workspace,
                        DeviceContext& device) {
    bounds_.assign(bounds.begin(), bounds.end());
    workspace_bytes_.resize(bounds.size());
    const auto cuda = device.execution_view();
    if (experts)
        prepare_offloaded(run, parameters, *experts, ple, output, logits, columns, workspace, cuda);
    else
        prepare_resident(run, parameters, ple, output, logits, workspace, cuda);
    // Qualify every child/layer update before accepting requests.
    for (const auto bound : bounds_) {
        select(bound);
        if (resident()) {
            std::get<Resident>(implementation_).main.executable.upload(cuda.stream());
        } else {
            for (auto& layer : std::get<Offloaded>(implementation_).layers)
                layer->upload(cuda.stream());
        }
        device.synchronize();
    }
    select(bounds_.front());
}

void TextGraph::prepare_resident(execution::TextRun& run, const execution::Parameters& p,
                                 std::array<ple::DeviceEmbedding, 2> inputs,
                                 execution::TextOutputs output, Tensor logits,
                                 MemoryArena& workspace, CudaExecutionView cuda) {
    auto& graph = implementation_.emplace<Resident>();
    for (const auto bound : bounds_) {
        run.set_visible_tokens(bound);
        workspace.reset_peak();
        auto& profile = graph.profiles.emplace_back();
        for (std::uint32_t i = 0; i < p.text.layers.size(); ++i) {
            if (!std::holds_alternative<execution::QsaParameters>(p.text.layers[i].mixer)) continue;
            profile.emplace_back().capture(cuda.stream(), [&] { run.local_mixer(i); });
        }
        workspace_bytes_[graph.profiles.size() - 1] = workspace.peak_used();
    }
    workspace.reset_peak();
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
        graph.ple[slot].capture(cuda.stream(), [&] {
            inputs[slot].ready.wait_external(cuda.stream());
            run.consume_ple(inputs[slot].values);
            inputs[slot].consumed.record_external(cuda.stream());
        });
    }
    run.set_visible_tokens(bounds_.front());
    graph.main.prepare(cuda, [&] {
        run.stem();
        std::size_t mixer = 0;
        for (std::uint32_t i = 0; i < p.text.layers.size(); ++i) {
            if (i == p.profile->config.text.ple.layer)
                graph.ple_node = graph.ple[0].append_to(cuda.stream());
            if (std::holds_alternative<execution::QsaParameters>(p.text.layers[i].mixer)) {
                graph.mixers.push_back(graph.profiles.front()[mixer++].append_to(cuda.stream()));
                run.finish_mixer(i);
            } else
                run.mixer(i);
            run.resident_moe(i);
            run.combine(i);
        }
        run.finish(output, logits);
    });
    for (auto& bytes : workspace_bytes_) bytes = std::max(bytes, workspace.peak_used());
}

void TextGraph::prepare_offloaded(execution::TextRun& run, const execution::Parameters& p,
                                  moe::Schedule& experts,
                                  std::array<ple::DeviceEmbedding, 2> inputs,
                                  execution::TextOutputs output, Tensor logits,
                                  std::int32_t columns, MemoryArena& workspace,
                                  CudaExecutionView cuda) {
    auto& graph        = implementation_.emplace<Offloaded>();
    graph.ple_layer    = p.profile->config.text.ple.layer;
    graph.block_output = run.block_output();
    workspace.reset_peak();
    graph.stem.prepare(cuda, [&] { run.stem(); });
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
        graph.ple[slot].prepare(cuda, [&] {
            inputs[slot].ready.wait_external(cuda.stream());
            run.consume_ple(inputs[slot].values);
            inputs[slot].consumed.record_external(cuda.stream());
        });
    }
    graph.finish.prepare(cuda, [&] { run.finish(output, logits); });
    const auto fixed_workspace = workspace.peak_used();
    for (const auto bound : bounds_) {
        run.set_visible_tokens(bound);
        workspace.reset_peak();
        auto& profile = graph.profiles.emplace_back();
        for (std::uint32_t i = 0; i < p.text.layers.size(); ++i) {
            const auto* shared = graph.profiles.size() == 1 ? nullptr : &graph.profiles.front()[i];
            if (shared &&
                std::holds_alternative<execution::GdnParameters>(p.text.layers[i].mixer)) {
                profile.push_back(*shared);
            } else {
                profile.push_back(experts.capture_layer(
                    p.text.layers[i].moe.bank, cuda, columns, [&] { run.mixer(i); },
                    [&](const moe::GpuCall& call) { run.gpu_moe(call); }, [&] { run.combine(i); },
                    shared ? *shared : nullptr));
            }
        }
        workspace_bytes_[graph.profiles.size() - 1] =
            std::max({fixed_workspace, workspace.peak_used(), workspace_bytes_.front()});
    }
    for (const auto& layer : graph.profiles.front()) graph.layers.push_back(layer->instantiate());
}

bool TextGraph::resident() const { return std::holds_alternative<Resident>(implementation_); }

void TextGraph::select(std::uint32_t visible) {
    const auto it = std::lower_bound(bounds_.begin(), bounds_.end(), visible);
    if (it == bounds_.end()) throw std::logic_error("Qwen4Exp graph context exceeds capacity");
    const auto next = static_cast<std::size_t>(it - bounds_.begin());
    if (next == installed_) return;
    if (resident()) {
        auto& graph = std::get<Resident>(implementation_);
        for (std::size_t i = 0; i < graph.mixers.size(); ++i)
            graph.main.executable.update_child(graph.mixers[i], graph.profiles[next][i]);
    } else {
        auto& graph = std::get<Offloaded>(implementation_);
        for (std::size_t i = 0; i < graph.layers.size(); ++i)
            if (graph.profiles[next][i] != graph.profiles[installed_][i])
                graph.layers[i]->update(*graph.profiles[next][i]);
    }
    installed_ = next;
}

void TextGraph::replay(std::uint32_t slot, CudaExecutionView cuda) {
    auto& graph = std::get<Resident>(implementation_);
    if (slot != graph.slot) {
        graph.main.executable.update_child(graph.ple_node, graph.ple.at(slot));
        graph.slot = slot;
    }
    graph.main.executable.launch(cuda.stream());
}

bool TextGraph::Offloaded::advance(std::uint32_t end, CudaExecutionView cuda) {
    while (next_layer < end) {
        const nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime, next_layer);
        auto& layer         = *layers[next_layer];
        const bool complete = waiting_layer ? layer.poll() : layer.start(block_output, cuda);
        if (!complete) {
            waiting_layer = true;
            return false;
        }
        waiting_layer = false;
        ++next_layer;
    }
    return true;
}

void TextGraph::begin_prefix(CudaExecutionView cuda) {
    auto& graph         = std::get<Offloaded>(implementation_);
    graph.next_layer    = 0;
    graph.waiting_layer = false;
    graph.stem.executable.launch(cuda.stream());
}

bool TextGraph::advance_prefix(CudaExecutionView cuda) {
    auto& graph = std::get<Offloaded>(implementation_);
    return graph.advance(graph.ple_layer, cuda);
}

void TextGraph::consume_ple(std::uint32_t slot, CudaExecutionView cuda) {
    if (slot >= 2) throw std::logic_error("invalid PLE graph slot");
    std::get<Offloaded>(implementation_).ple[slot].executable.launch(cuda.stream());
}

bool TextGraph::advance_suffix(CudaExecutionView cuda) {
    auto& graph = std::get<Offloaded>(implementation_);
    if (!graph.advance(graph.layers.size(), cuda)) return false;
    graph.finish.executable.launch(cuda.stream());
    return true;
}

void TextGraph::cancel() {
    if (auto* graph = std::get_if<Offloaded>(&implementation_)) {
        if (graph->waiting_layer) graph->layers.at(graph->next_layer)->cancel();
        graph->waiting_layer = false;
        graph->next_layer    = 0;
    }
}

} // namespace ninfer::models::qwen4_exp
