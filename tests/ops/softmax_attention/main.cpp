#include "causal/fixture.h"

void add_plain_attention_cases(ninfer::test::OpSuite&);
void add_context_attention_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    ninfer::test::causal_attention::add_cases(suite);
    add_plain_attention_cases(suite);
    add_context_attention_cases(suite);
    return suite.main(argc, argv);
}
