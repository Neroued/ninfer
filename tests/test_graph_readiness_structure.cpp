#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string read_file(const char* path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error(std::string("failed to open ") + path); }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool contains(const std::string& haystack, const char* needle) {
    return haystack.find(needle) != std::string::npos;
}

int expect_absent(const std::string& text, const char* needle, const char* message) {
    if (!contains(text, needle)) { return 0; }
    std::cerr << message << ": found `" << needle << "`\n";
    return 1;
}

int expect_present(const std::string& text, const char* needle, const char* message) {
    if (contains(text, needle)) { return 0; }
    std::cerr << message << ": missing `" << needle << "`\n";
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    const std::string model = read_file(QUS_SOURCE_DIR "/src/model/qwen3_6_27b.cpp");
    failures += expect_absent(model, "cudaMemcpyAsync(io_.pos.data",
                              "decode/prefill position must be set on device");
    failures +=
        expect_absent(model, "pos_upload_", "decode position must not use a host scalar cache");
    failures += expect_present(model, "detail::fill_positions(positions, s)",
                               "prefill must initialize positions on device");
    failures +=
        expect_present(model, "detail::set_pos(io_.pos, T, s)", "prefill must set device pos");
    failures +=
        expect_present(model, "detail::advance_pos(io_.pos, s)", "decode must advance device pos");
    failures += expect_absent(model, "gdn_" "conv1d_storage_",
                              "canonical conv1d must not have private model storage");
    failures += expect_absent(model, "cudaMalloc(&" "storage",
                              "canonical conv1d bind must not allocate device storage");
    failures += expect_absent(model, "unexpected q5090 conv1d shape",
                              "canonical conv1d errors must name the required shape");
    failures += expect_present(model, "expected q5090 canonical conv1d shape [10240,4,1]",
                               "canonical conv1d bind must document the required shape");

    const std::string position = read_file(QUS_SOURCE_DIR "/src/model/position.cu");
    failures += expect_present(position, "set_pos_kernel<<<1, 1, 0, stream>>>",
                               "set_pos must use a one-thread device kernel");
    failures += expect_present(position, "advance_pos_kernel<<<1, 1, 0, stream>>>",
                               "advance_pos must use a one-thread device kernel");

    const std::string gdr_header =
        read_file(QUS_SOURCE_DIR "/include/qus/kernels/gated_delta_rule.h");
    failures += expect_present(gdr_header, "float scale, WorkspaceArena& ws,",
                               "gated_delta_rule_recurrent API must accept WorkspaceArena");
    const std::string gdr = read_file(QUS_SOURCE_DIR "/src/kernels/wrapper/gated_delta_rule.cpp");
    failures += expect_absent(gdr, "thread_local",
                              "gated_delta_rule_recurrent scratch must come from WorkspaceArena");
    failures += expect_absent(gdr, "cudaMalloc(&ptr",
                              "gated_delta_rule_recurrent wrapper must not allocate scratch");
    failures += expect_absent(gdr, "recurrent_scratch_for",
                              "gated_delta_rule_recurrent wrapper must not own scratch state");
    failures += expect_present(gdr, "float scale, WorkspaceArena& ws,",
                               "gated_delta_rule_recurrent wrapper must accept WorkspaceArena");
    failures += expect_present(gdr, "ArenaScope arena_scope(ws)",
                               "gated_delta_rule_recurrent must scope arena scratch");
    failures += expect_present(gdr, "ws.alloc(DType::FP32",
                               "gated_delta_rule_recurrent must allocate FP32 scratch from arena");

    if (failures != 0) {
        std::cerr << "FAIL graph readiness structural invariants\n";
        return 1;
    }

    std::cout << "OK graph readiness structural invariants\n";
    return 0;
}
