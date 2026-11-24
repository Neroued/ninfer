#include "ops/linear/split/runner.h"

int main(int argc, char** argv) {
    using namespace ninfer::test::linear_split;
    Suite suite;
    add_segmented_cases(suite);
    add_two_output_cases(suite);
    add_multi_output_cases(suite);
    add_bank_cases(suite);
    return suite.main(argc, argv);
}
