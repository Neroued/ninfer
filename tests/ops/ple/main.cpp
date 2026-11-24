#include "ops/suite.h"

void add_ple_gate_cases(ninfer::test::OpSuite&);
void add_ple_conv_cases(ninfer::test::OpSuite&);
void add_ple_state_extract_cases(ninfer::test::OpSuite&);
void add_ple_lookup_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_ple_lookup_cases(suite);
    add_ple_gate_cases(suite);
    add_ple_conv_cases(suite);
    add_ple_state_extract_cases(suite);
    return suite.main(argc, argv);
}
