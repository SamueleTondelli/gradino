#include "include/utils.h"
#include "include/test.h"

int main() {
    init_random();

    test_add(1024, 1024);
    test_mul(512, 512, 512);
    test_reduce_add(128, 128, 2);
    test_arena(GiB(4), MiB(1), KiB(500), 100);
    test_grad_relu();
}
