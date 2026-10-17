#include "ops/suite.h"

void add_add_bias_cases(ninfer::test::OpSuite&);
void add_gelu_cases(ninfer::test::OpSuite&);
void add_silu_mul_cases(ninfer::test::OpSuite&);
void add_residual_add_cases(ninfer::test::OpSuite&);
void add_sum_contributions_cases(ninfer::test::OpSuite&);
void add_sigmoid_mul_cases(ninfer::test::OpSuite&);
void add_scalar_cases(ninfer::test::OpSuite&);
void add_cast_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_add_bias_cases(suite);
    add_gelu_cases(suite);
    add_silu_mul_cases(suite);
    add_residual_add_cases(suite);
    add_sum_contributions_cases(suite);
    add_sigmoid_mul_cases(suite);
    add_scalar_cases(suite);
    add_cast_cases(suite);
    return suite.main(argc, argv);
}
