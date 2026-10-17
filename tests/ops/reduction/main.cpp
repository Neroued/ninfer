#include "ops/suite.h"

void add_argmax_cases(ninfer::test::OpSuite&);
void add_target_logprobs_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_argmax_cases(suite);
    add_target_logprobs_cases(suite);
    return suite.main(argc, argv);
}
