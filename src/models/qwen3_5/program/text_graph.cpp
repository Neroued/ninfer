#include "models/qwen3_5/program/text_graph.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
TextGraphDefinition::TextGraphDefinition(std::size_t count, const TextGraphDefinition* shared)
    : layers(count) {
    if (shared) layers = shared->layers;
}

struct TextGraphExecutable::Target {
    std::vector<std::unique_ptr<moe::LayerGraphExecutable>> layers;
    const TextGraphDefinition* installed;

    explicit Target(const TextGraphDefinition& definition) : installed(&definition) {
        layers.reserve(definition.layers.size());
        for (const auto& layer : definition.layers) layers.push_back(layer->instantiate());
    }
};

TextGraphExecutable::TextGraphExecutable(const TextGraphDefinition& definition,
                                         const TextGraphExecutable* shared)
    : target_(shared ? shared->target_ : std::make_shared<Target>(definition)) {
    prefix_.instantiate(definition.prefix);
    suffix_.instantiate(definition.suffix);
}

void TextGraphExecutable::update(const TextGraphDefinition& definition) {
    prefix_.update(definition.prefix);
    suffix_.update(definition.suffix);
    activate(definition);
}

void TextGraphExecutable::activate(const TextGraphDefinition& definition) {
    if (target_->installed == &definition) return;
    if (target_->layers.size() != definition.layers.size())
        throw std::logic_error("text graph layer count changed");
    for (std::size_t i = 0; i < target_->layers.size(); ++i)
        target_->layers[i]->update(*definition.layers[i]);
    target_->installed = &definition;
}

void TextGraphExecutable::upload(cudaStream_t stream) {
    prefix_.upload(stream);
    suffix_.upload(stream);
    for (auto& layer : target_->layers) layer->upload(stream);
}

TextGraphRun::TextGraphRun(TextGraphDefinition& definition, CudaExecutionView cuda)
    : definition_(&definition), cuda_(cuda) {}

TextGraphRun::TextGraphRun(TextGraphExecutable& executable, CudaExecutionView cuda,
                           const std::function<void()>* after_first_submit)
    : executable_(&executable), cuda_(cuda), after_first_submit_(after_first_submit) {}

void TextGraphRun::begin() {
    if (recording()) capture_.emplace(cuda_.stream());
}

void TextGraphRun::begin_layers() {
    if (traversed_) throw std::logic_error("text graph has more than one target traversal");
    traversed_ = true;
    if (recording()) {
        definition_->prefix = capture_->finish();
        capture_.reset();
    } else {
        executable_->prefix_.launch(cuda_.stream());
    }
}

void TextGraphRun::end_layers() {
    if (recording()) capture_.emplace(cuda_.stream());
}

void TextGraphRun::finish() {
    if (!traversed_) throw std::logic_error("text graph did not execute its target traversal");
    if (recording()) {
        definition_->suffix = capture_->finish();
        capture_.reset();
    } else {
        executable_->suffix_.launch(cuda_.stream());
    }
}

void TextGraphRun::first_submitted() {
    if (!notified_) {
        notified_ = true;
        if (after_first_submit_) (*after_first_submit_)();
    }
}

void TextGraphRun::capture_layer(std::size_t layer, moe::Schedule& resources, ExpertBankId bank,
                                 std::int32_t tokens, const std::function<void()>& mixer,
                                 const std::function<void(const moe::GpuCall&)>& gpu_moe,
                                 const std::function<void()>& feature) {
    auto& definition = definition_->layers.at(layer);
    definition = resources.capture_layer(bank, cuda_, tokens, mixer, gpu_moe, feature, definition);
}

void TextGraphRun::replay_layer(std::size_t layer, Tensor& output) {
    executable_->target_->layers.at(layer)->replay(output, cuda_, [this] { first_submitted(); });
}

} // namespace ninfer::models::qwen3_5::execution
