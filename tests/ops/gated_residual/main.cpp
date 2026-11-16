#include "support.h"

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    suite.filter("width");
    ninfer::test::hc::add_norm_cases(suite);
    ninfer::test::hc::add_bf16_mix_cases(suite);
    ninfer::test::hc::add_fp8_mix_cases(suite);
    return suite.main(argc, argv);
}
