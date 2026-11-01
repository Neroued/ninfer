#include "ops/suite.h"

void add_sampling_cases(ninfer::test::OpSuite&);
void add_candidate_selector_cases(ninfer::test::OpSuite&);
void add_greedy_accept_cases(ninfer::test::OpSuite&);
void add_sparse_accept_cases(ninfer::test::OpSuite&);

void add_vocabulary_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_sampling_cases(suite);
    add_vocabulary_cases(suite);
    add_candidate_selector_cases(suite);
    add_greedy_accept_cases(suite);
    add_sparse_accept_cases(suite);
    return suite.main(argc, argv);
}
