#include "ops/suite.h"

void add_rope_cases(ninfer::test::OpSuite&);
void add_rmsnorm_rope_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_rope_cases(suite);
    add_rmsnorm_rope_cases(suite);
    return suite.main(argc, argv);
}
