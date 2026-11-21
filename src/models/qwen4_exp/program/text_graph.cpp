#include "models/qwen4_exp/program/text_graph.h"
#include "core/nvtx.h"
#include "models/common/parallel_execution.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

void QsaGraphReadout::prepare(execution::QsaRun run, std::span<const std::uint32_t> bounds,
                              CudaExecutionView cuda) {
    const auto call = run.readout(bounds);
    query_          = call.query;
    output_         = call.output;
    for (const auto visible : bounds)
        profiles_.emplace_back().capture(cuda.stream(), [&] { call.execute(visible, cuda); });
}

DecodeGraphChild QsaGraphReadout::append(const execution::QsaReadout& call,
                                         cudaStream_t stream) const {
    if (call.query.data != query_.data || call.output.data != output_.data)
        throw std::logic_error("QSA graph readout binding differs from execution layout");
    return profiles_.front().append_to(stream);
}

void TextGraph::prepare(execution::TextRun& run, const execution::Parameters& parameters,
                        moe::Schedule* experts, std::span<const std::uint32_t> bounds,
                        std::array<ple::DeviceEmbedding, 2> ple, execution::TextOutputs output,
                        Tensor logits, moe::TokenGeometry geometry, MemoryArena& workspace,
                        DeviceContext& device, const std::function<void()>& install_inputs) {
    bounds_.assign(bounds.begin(), bounds.end());
    const auto cuda = device.execution_view();
    workspace.reset_peak();
    readouts_.resize(parameters.text.layers.size());
    for (std::size_t i = 0; i < readouts_.size(); ++i)
        if (std::holds_alternative<execution::QsaParameters>(parameters.text.layers[i].mixer))
            readouts_[i].emplace().prepare(run.local_qsa(i), bounds, cuda);
    workspace_bytes_ = workspace.peak_used();
    if (experts)
        prepare_offloaded(run, parameters, *experts, ple, output, logits, geometry, workspace, cuda,
                          install_inputs);
    else
        prepare_resident(run, parameters, ple, output, logits, workspace, cuda, install_inputs);
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
                                 MemoryArena& workspace, CudaExecutionView cuda,
                                 const std::function<void()>& install_inputs) {
    auto& graph = implementation_.emplace<Resident>();
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
        install_inputs();
        run.stem();
        graph.mixers.resize(readouts_.size());
        for (std::uint32_t i = 0; i < p.text.layers.size(); ++i) {
            if (i == p.profile->config.text.ple.layer)
                graph.ple_node = graph.ple[0].append_to(cuda.stream());
            run.mixer(i, [&](const execution::QsaReadout& call) {
                graph.mixers[i] = readouts_[i]->append(call, cuda.stream());
            });
            run.resident_moe(i);
            run.combine(i);
        }
        capture_finish(run, p, output, logits, workspace, cuda);
    });
    workspace_bytes_ = std::max(workspace_bytes_, workspace.peak_used());
}

void TextGraph::prepare_offloaded(execution::TextRun& run, const execution::Parameters& p,
                                  moe::Schedule& experts,
                                  std::array<ple::DeviceEmbedding, 2> inputs,
                                  execution::TextOutputs output, Tensor logits,
                                  moe::TokenGeometry geometry, MemoryArena& workspace,
                                  CudaExecutionView cuda,
                                  const std::function<void()>& install_inputs) {
    auto& graph        = implementation_.emplace<Offloaded>();
    graph.ple_layer    = p.profile->config.text.ple.layer;
    graph.block_output = run.block_output();
    workspace.reset_peak();
    graph.stem.prepare(cuda, [&] {
        install_inputs();
        run.stem();
    });
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
        graph.ple[slot].prepare(cuda, [&] {
            inputs[slot].ready.wait_external(cuda.stream());
            run.consume_ple(inputs[slot].values);
            inputs[slot].consumed.record_external(cuda.stream());
        });
    }
    graph.finish.prepare(cuda, [&] { capture_finish(run, p, output, logits, workspace, cuda); });
    for (std::uint32_t i = 0; i < p.text.layers.size(); ++i) {
        auto definition = experts.capture_layer(
            p.text.layers[i].moe.bank, cuda, geometry,
            [&](DecodeGraphChild& child) {
                run.mixer(i, [&](const execution::QsaReadout& call) {
                    child = readouts_[i]->append(call, cuda.stream());
                });
            },
            [&](const moe::GpuCall& call) { run.gpu_moe(call); }, [&] { run.combine(i); }, nullptr);
        graph.layers.push_back(definition->instantiate());
        graph.definitions.push_back(std::move(definition));
    }
    workspace_bytes_ = std::max(workspace_bytes_, workspace.peak_used());
}

void TextGraph::capture_finish(execution::TextRun& run, const execution::Parameters& p,
                               execution::TextOutputs output, Tensor logits, MemoryArena& workspace,
                               CudaExecutionView cuda) {
    run.finish(output);
    head_.capture(cuda.stream(), [&] {
        const int columns = logits.numel() / logits.ne[0];
        auto hidden       = output.sample.view({dimension(p.profile->config.text.hidden), columns});
        auto projected    = logits.view({logits.ne[0], columns});
        execution::project_head(hidden, p.text.head, projected, workspace, cuda);
    });
}

void TextGraph::set_head(bool enabled) {
    auto& executable = resident() ? std::get<Resident>(implementation_).main.executable
                                  : std::get<Offloaded>(implementation_).finish.executable;
    head_.set_enabled(executable, enabled);
}

bool TextGraph::resident() const { return std::holds_alternative<Resident>(implementation_); }

void TextGraph::select(std::uint32_t visible) {
    const auto it = std::lower_bound(bounds_.begin(), bounds_.end(), visible);
    if (it == bounds_.end()) throw std::logic_error("Qwen4Exp graph context exceeds capacity");
    const auto next = static_cast<std::size_t>(it - bounds_.begin());
    if (next == installed_) return;
    if (resident()) {
        auto& graph = std::get<Resident>(implementation_);
        for (std::size_t i = 0; i < readouts_.size(); ++i)
            if (readouts_[i])
                graph.main.executable.update_child(graph.mixers[i], readouts_[i]->at(next));
    } else {
        auto& graph = std::get<Offloaded>(implementation_);
        for (std::size_t i = 0; i < readouts_.size(); ++i)
            if (readouts_[i]) graph.layers[i]->update_mixer_child(readouts_[i]->at(next));
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
