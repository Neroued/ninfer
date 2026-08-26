#include <iostream>

int run_softmax_attention_causal_cache_tests();
int run_softmax_attention_plain_and_packed_tests();
int run_softmax_attention_context_tests();

int main() {
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
