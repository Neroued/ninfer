// qus — Qwen3.6-27B single-stream inference bench driver (stub).
// token-ids in -> token-ids out (greedy, v1). See docs/design.md.
#include "qus/runtime/engine.h"

int main(int /*argc*/, char** /*argv*/) {
    // TODO(impl): parse args; Engine.load(weight_file); prefill(prompt_ids);
    //             decode loop -> emit token ids; report tok/s.
    return 0;
}
