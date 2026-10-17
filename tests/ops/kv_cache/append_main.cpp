#include "ops/suite.h"

void add_full_append_cases(ninfer::test::OpSuite&);
void add_prefix_append_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_full_append_cases(suite);
    add_prefix_append_cases(suite);
    return suite.main(argc, argv);
}
