#include "ops/linear/projection/runner.h"

int main(int argc, char** argv) {
    using namespace ninfer::test::linear;
    Suite suite;
    add_bf16_cases(suite);
    add_q4_cases(suite);
    add_q5_cases(suite);
    add_q6_cases(suite);
    add_q8_cases(suite);
    add_fp8_cases(suite);
    add_nvfp4_cases(suite);
    add_bank_cases(suite);
    return suite.main(argc, argv);
}
