#include "ops/suite.h"

void add_conv_prepare_cases(ninfer::test::OpSuite&);
void add_linear_conv_add_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_conv_prepare_cases(suite);
    add_linear_conv_add_cases(suite);
    return suite.main(argc, argv);
}
