#include "ninfer/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<ninfer::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<ninfer::PreparedPrompt>);
static_assert(std::is_move_constructible_v<ninfer::Engine>);
static_assert(!std::is_copy_constructible_v<ninfer::Engine>);

int main() {
    ninfer::EngineOptions options;
    options.artifact_path = "model.ninfer";
    return options.artifact_path.extension() == ".ninfer" ? 0 : 1;
}
