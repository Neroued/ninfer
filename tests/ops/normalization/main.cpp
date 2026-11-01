#include "ops/suite.h"

void add_l2norm_cases(ninfer::test::OpSuite&);
void add_layer_norm_cases(ninfer::test::OpSuite&);
void add_rmsnorm_pack_tail_cases(ninfer::test::OpSuite&);
void add_rmsnorm_cases(ninfer::test::OpSuite&);
void add_gated_rmsnorm_cases(ninfer::test::OpSuite&);
void add_fused_rmsnorm_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_l2norm_cases(suite);
    add_layer_norm_cases(suite);
    add_rmsnorm_pack_tail_cases(suite);
    add_rmsnorm_cases(suite);
    add_gated_rmsnorm_cases(suite);
    add_fused_rmsnorm_cases(suite);
    return suite.main(argc, argv);
}
