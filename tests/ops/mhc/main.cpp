#include "support.h"

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    ninfer::test::mhc::add_cases(suite);
    return suite.main(argc, argv);
}
