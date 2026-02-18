#include "include/utils.h"
#include "include/test.h"

int main() {
    init_random();

    // test_add(1024, 1024);
    // test_mul(512, 512, 512);
    // test_reduce_add(128, 128, 2);
    // test_arena(GiB(4), MiB(1), KiB(500), 100);
    // test_grad_relu();
    // test_grad_bwd();
    // test_xor(1e-1, 0.0, 16, 20);
    // test_bwd_perf(5e-3, 256, 128, 256, 64, 10);
    // test_split_views();
    // test_concat();
    test_lstm(32, 64, 64, 20, 16, 10);
}
