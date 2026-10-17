#include "ops/suite.h"

void add_scatter_cases(ninfer::test::OpSuite&);
void add_scatter_batch_cases(ninfer::test::OpSuite&);
void add_position_cases(ninfer::test::OpSuite&);
void add_features_cases(ninfer::test::OpSuite&);

void add_gather_remap_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_scatter_cases(suite);
    add_scatter_batch_cases(suite);
    add_position_cases(suite);
    add_features_cases(suite);
    add_gather_remap_cases(suite);
    return suite.main(argc, argv);
}
