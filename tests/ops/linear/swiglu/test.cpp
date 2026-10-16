#include "ops/linear/swiglu/runner.h"

int main(int argc, char** argv) {
    using namespace ninfer::test::linear_swiglu;
    Suite suite;
    add_q4_cases(suite);
    add_q8_cases(suite);
    add_nvfp4_cases(suite);
    add_fp8_cases(suite);
    add_bank_cases(suite);
    return suite.main(argc, argv);
}
