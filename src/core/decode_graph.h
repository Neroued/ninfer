#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace ninfer {

// A child node belongs to its enclosing definition. Keep that definition alive while
// updating the executable instantiated from it.
struct DecodeGraphChild {
    cudaGraphNode_t node = nullptr;
};

class DecodeGraphExecutable;

// Inline kernel/copy/memset nodes in one parent capture. The enclosing definition and
// executable outlive the section; recapture the section when replacing that executable.
class DecodeGraphSection {
public:
    DecodeGraphSection()                                         = default;
    DecodeGraphSection(const DecodeGraphSection&)                = delete;
    DecodeGraphSection& operator=(const DecodeGraphSection&)     = delete;
    DecodeGraphSection(DecodeGraphSection&&) noexcept            = default;
    DecodeGraphSection& operator=(DecodeGraphSection&&) noexcept = default;

    void capture(cudaStream_t, const std::function<void()>&);
    void set_enabled(const DecodeGraphExecutable&, bool);

private:
    std::vector<cudaGraphNode_t> nodes_;
    bool enabled_ = true;
};

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    [[nodiscard]] DecodeGraphChild append_to(cudaStream_t capturing_stream) const;
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphCapture;
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

// An unfinished capture is discarded on scope exit. All participating streams must
// rejoin the origin before finish(); the definition owns the resulting graph.
class DecodeGraphCapture {
public:
    explicit DecodeGraphCapture(cudaStream_t stream);
    ~DecodeGraphCapture();
    DecodeGraphCapture(const DecodeGraphCapture&)            = delete;
    DecodeGraphCapture& operator=(const DecodeGraphCapture&) = delete;

    [[nodiscard]] DecodeGraphDefinition finish();

private:
    cudaStream_t stream_;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void update_child(DecodeGraphChild child, const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphSection;
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace ninfer
