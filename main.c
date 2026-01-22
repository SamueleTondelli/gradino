#include "include/arena.h"
#include "include/tensor.h"
#include "include/utils.h"
#include <math.h>
#include <stdio.h>

void add_test() {
    u32 a_shape[] = {1, 1, 1024, 1024};
    Tensor* a = create_tensor(a_shape, 4);
    randomize_tensor(a, 0.0, 1.0);

    u32 b_shape[] = {1, 1, 1024, 1024};
    Tensor* b = create_tensor(b_shape, 4);
    randomize_tensor(b, 0.0, 1.0);
    
    print_tensor(a, false);
    print_tensor(b, false);


    double start = perf_counter_ns();
    Tensor* c = add_tensor(a, b);
    double elapsed = perf_counter_ns() - start;
    printf("Vectorized took %.3fms\n", elapsed/1000000.0);


    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
}

static void matmul(const f32* a, const f32* b, f32* res, u32 a_rows, u32 a_cols, u32 b_cols) {
    for (u32 i = 0; i < a_rows; i++) {
        for (u32 j = 0; j < b_cols; j++) {
            f32 el = 0.0;
            for (u32 k = 0; k < a_cols; k++) {
                el += a[i * a_cols + k] * b[k * b_cols + j];
            }
            res[i * b_cols + j] = el;
        }
    }
}

void mul_test() {
    u32 size = 512;
    u32 a_shape[] = {1, 1, size, size};
    Tensor* a = create_tensor(a_shape, 4);

    u32 i_shape[] = {1, 1, size, size};
    Tensor* b = create_tensor(i_shape, 4);

    randomize_tensor(a, 0.0, 1.0);
    randomize_tensor(b, 0.0, 1.0);

    double start = perf_counter_ns();
    Tensor* c = mul_tensor(a, b);
    double elapsed = perf_counter_ns() - start;
    print_tensor(c, false);
    printf("Vec Multiplication took %.3fms\n", elapsed/1000000.0);

    f32* res = malloc(size * size * sizeof(f32));
    matmul(a->data, b->data, res, size, size, size);
    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            if (fabsf(c->data[i * size + j] - res[i * size + j]) > 1e-6) {
                printf("Wrong element at (%u, %u): %f, %f\n", i, j, c->data[i * size + j], res[i * size + j]);
            }
        }
    }

    free(res);
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
}

void arena_test() {
    arena_allocator* arena = create_arena(GiB(4), MiB(1), 8);

    usize size = KiB(500);
    for (int i = 0; i < 100; i++) {
        // getchar();
        printf("Allocating %lu bytes\n", size);
        u8* mem = alloc_arena(arena, size, 1); 
        if (mem == NULL) {
            printf("Failed to alloc\n");
        }
        mem[0] = 1;
    }

    printf("Allocated %x bytes\n", arena->alloc_pos);
    
    free_arena(arena);
    if (!destroy_arena(arena)) {
        printf("Failed to destroy arena\n");
    }
}

int main() {
    init_random();
    // add_test();    
    mul_test();
    // arena_test();
}
