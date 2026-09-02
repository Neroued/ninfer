#include <iostream>
#include <string_view>

int run_softmax_attention_causal_cache_tests();
int run_softmax_attention_nvfp4_tests();
int run_softmax_attention_k8v4_tests();
int run_softmax_attention_plain_and_packed_tests();
int run_softmax_attention_context_tests();

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--nvfp4-only") {
        return run_softmax_attention_nvfp4_tests();
    }
    if (argc == 2 && std::string_view(argv[1]) == "--k8v4-only") {
        return run_softmax_attention_k8v4_tests();
    }
    if (argc != 1) {
        std::cerr << "usage: ninfer_softmax_attention_test [--nvfp4-only|--k8v4-only]\n";
        return 2;
    }
    const int causal = run_softmax_attention_causal_cache_tests();
    if (causal == 77) return 77;

    const int plain_and_packed = run_softmax_attention_plain_and_packed_tests();
    if (plain_and_packed == 77) return 77;

    const int context = run_softmax_attention_context_tests();
    if (context == 77) return 77;

    const int failures = causal + plain_and_packed + context;
    std::cout << (failures == 0 ? "softmax_attention: PASS\n" : "softmax_attention: FAIL\n");
    return failures == 0 ? 0 : 1;
}
