#include "ops/suite.h"

void add_ragged_prefix_cases(ninfer::test::OpSuite&);
void add_masked_block_cases(ninfer::test::OpSuite&);
void add_autoregressive_round_cases(ninfer::test::OpSuite&);

void add_verify_cases(ninfer::test::OpSuite&);

void add_causal_pairs_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_ragged_prefix_cases(suite);
    add_masked_block_cases(suite);
    add_autoregressive_round_cases(suite);
    add_verify_cases(suite);
    add_causal_pairs_cases(suite);
    return suite.main(argc, argv);
}
