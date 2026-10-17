#include "support.h"

void add_control_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_control_cases(suite);
    ninfer::test::gdn_control::add_projection_cases(suite);
    return suite.main(argc, argv);
}
