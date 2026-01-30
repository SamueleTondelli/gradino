#include "include/arena.h"
#include "include/tensor.h"
#include "include/utils.h"
#include "include/grad.h"
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
    u32 size = 256;
    u32 a_shape[] = {1, 1, size, size};
    Tensor* a = create_tensor(a_shape, 4);

    u32 b_shape[] = {1, 1, size, size};
    Tensor* b = create_tensor(b_shape, 4);

    randomize_tensor(a, 0.0, 1.0);
    randomize_tensor(b, 0.0, 1.0);

    double start = perf_counter_ns();
    Tensor* c = mul_tensor_tr(a, b, false, false);
    double elapsed = perf_counter_ns() - start;
    print_tensor(c, false);
    printf("Vec Multiplication took %.3fms\n", elapsed/1000000.0);

    f32* res = malloc(size * size * sizeof(f32));
    matmul(a->data, b->data, res, size, size, size);
    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            if (fabsf(c->data[i * size + j] - res[i * size + j]) > 1e-4) {
                printf("Wrong element at (%u, %u): %f, %f\n", i, j, c->data[i * size + j], res[i * size + j]);
            }
        }
    }

    free(res);
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
}

static void matmul_at(const f32* a, const f32* b, f32* res, u32 a_rows, u32 a_cols, u32 b_cols) {
    for (u32 i = 0; i < a_rows; i++) {
        for (u32 j = 0; j < b_cols; j++) {
            f32 el = 0.0;
            for (u32 k = 0; k < a_cols; k++) {
                el += a[k * a_rows + i] * b[k * b_cols + j];
            }
            res[i * b_cols + j] = el;
        }
    }
}

static void matmul_bt(const f32* a, const f32* b, f32* res, u32 a_rows, u32 a_cols, u32 b_rows) {
    for (u32 i = 0; i < a_rows; i++) {
        for (u32 j = 0; j < b_rows; j++) {
            f32 el = 0.0;
            for (u32 k = 0; k < a_cols; k++) {
                el += a[i * a_cols + k] * b[j * a_cols + k];
            }
            res[i * b_rows + j] = el;
        }
    }
}

static void matmul_atbt(const f32* a, const f32* b, f32* res, u32 a_rows, u32 a_cols, u32 b_rows) {
    for (u32 i = 0; i < a_rows; i++) {
        for (u32 j = 0; j < b_rows; j++) {
            f32 el = 0.0;
            for (u32 k = 0; k < a_cols; k++) {
                el += a[k * a_rows + i] * b[j * a_cols + k];
            }
            res[i * b_rows + j] = el;
        }
    }
}

void mul_bt_test() {
    u32 size = 512;
    u32 a_shape[] = {1, 1, size, size/2};
    Tensor* a = create_tensor(a_shape, 4);

    u32 b_shape[] = {1, 1, size, size/2};
    Tensor* b = create_tensor(b_shape, 4);

    randomize_tensor(a, 0.0, 1.0);
    randomize_tensor(b, 0.0, 1.0);

    double start = perf_counter_ns();
    Tensor* c = mul_tensor_tr(a, b, false, true);
    double elapsed = perf_counter_ns() - start;
    print_tensor(c, false);
    printf("Vec Multiplication Bt took %.3fms\n", elapsed/1000000.0);
    
    f32* res = malloc(size * size * sizeof(f32));
    matmul_bt(a->data, b->data, res, size, size/2, size);
    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            if (fabsf(c->data[i * size + j] - res[i * size + j]) > 1e-4) {
                printf("Wrong element at (%u, %u): %f, %f\n", i, j, c->data[i * size + j], res[i * size + j]);
                return;
            }
        }
    }

    free(res);
    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
}

void mul_at_test() {
    u32 size = 512;
    u32 a_shape[] = {1, 1, size/2, size};
    Tensor* a = create_tensor(a_shape, 4);

    u32 b_shape[] = {1, 1, size/2, size};
    Tensor* b = create_tensor(b_shape, 4);

    randomize_tensor(a, 0.0, 1.0);
    randomize_tensor(b, 0.0, 1.0);

    double start = perf_counter_ns();
    Tensor* c = mul_tensor_tr(a, b, true, false);
    double elapsed = perf_counter_ns() - start;
    print_tensor(c, false);
    printf("Vec Multiplication At took %.3fms\n", elapsed/1000000.0);
    
    f32* res = malloc(size * size * sizeof(f32));
    matmul_at(a->data, b->data, res, size, size/2, size);
    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            if (fabsf(c->data[i * size + j] - res[i * size + j]) > 1e-4) {
                printf("Wrong element at (%u, %u): %f, %f\n", i, j, c->data[i * size + j], res[i * size + j]);
                return;
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

void grad_test() {
    u32 shape[4] = {1, 1, 2, 2};
    GradTensor* gt = create_gradt(shape, 4);
    gt->tens->data[0] = 1.0;
    gt->tens->data[1] = 0.1;
    gt->tens->data[2] = -1;
    gt->tens->data[3] = 3.0;

    GradTensor* rgt = relu(gt);

    print_tensor(gt->tens, true);
    print_tensor(rgt->tens, true);
    
    free_gradt(gt);
    free(rgt);
}

int main() {
    init_random();
    // add_test();    
    // mul_test();
    // mul_bt_test();
    mul_at_test();
    // arena_test();
    // grad_test();
}
