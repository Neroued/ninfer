#pragma once

namespace ninfer::runtime {

template <class Program>
class GenerationGuard {
public:
    explicit GenerationGuard(Program& program) noexcept : program_(&program) {}

    ~GenerationGuard() noexcept {
        if (armed_) { program_->abort_request(); }
    }

    GenerationGuard(const GenerationGuard&)            = delete;
    GenerationGuard& operator=(const GenerationGuard&) = delete;

    void arm() noexcept { armed_ = true; }

    void complete() noexcept { armed_ = false; }

private:
    Program* program_ = nullptr;
    bool armed_       = false;
};

} // namespace ninfer::runtime
