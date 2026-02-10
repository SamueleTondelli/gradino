#include "include/arena.h"
#include "include/tensor.h"
#include "include/utils.h"
#include "include/grad.h"
#include <math.h>
#include <stdio.h>

void add_test() {
    u32 a_shape[] = {1, 1, 1024, 1024};
    Tensor* a = tensor_create(a_shape, 4);
    tensor_randomize(a, 0.0, 1.0);

    u32 b_shape[] = {1, 1, 1024, 1024};
    Tensor* b = tensor_create(b_shape, 4);
    tensor_randomize(b, 0.0, 1.0);
    
    tensor_print(a, false);
    tensor_print(b, false);


    double start = perf_counter_ns();
    Tensor* c = tensor_add(a, b);
    double elapsed = perf_counter_ns() - start;
    printf("Vectorized took %.3fms\n", elapsed/1000000.0);


    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
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
    Tensor* a = tensor_create(a_shape, 4);

    u32 b_shape[] = {1, 1, size, size};
    Tensor* b = tensor_create(b_shape, 4);

    tensor_randomize(a, 0.0, 1.0);
    tensor_randomize(b, 0.0, 1.0);

    double start = perf_counter_ns();
    Tensor* c = tensor_mul_tr(a, b, false, false);
    double elapsed = perf_counter_ns() - start;
    tensor_print(c, false);
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
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
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
    Tensor* a = tensor_create(a_shape, 4);

    u32 b_shape[] = {1, 1, size, size/2};
    Tensor* b = tensor_create(b_shape, 4);

    tensor_randomize(a, 0.0, 1.0);
    tensor_randomize(b, 0.0, 1.0);

    double start = perf_counter_ns();
    Tensor* c = tensor_mul_tr(a, b, false, true);
    double elapsed = perf_counter_ns() - start;
    tensor_print(c, false);
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
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void mul_at_test() {
    u32 size = 512;
    u32 a_shape[] = {1, 1, size/2, size};
    Tensor* a = tensor_create(a_shape, 4);

    u32 b_shape[] = {1, 1, size/2, size};
    Tensor* b = tensor_create(b_shape, 4);

    tensor_randomize(a, 0.0, 1.0);
    tensor_randomize(b, 0.0, 1.0);

    double start = perf_counter_ns();
    Tensor* c = tensor_mul_tr(a, b, true, false);
    double elapsed = perf_counter_ns() - start;
    tensor_print(c, false);
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
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void arena_test() {
    arena_allocator* arena = arena_create(GiB(4), MiB(1), 8);

    usize size = KiB(500);
    for (int i = 0; i < 100; i++) {
        // getchar();
        printf("Allocating %lu bytes\n", size);
        u8* mem = arena_alloc(arena, size, 1); 
        if (mem == NULL) {
            printf("Failed to alloc\n");
        }
        mem[0] = 1;
    }

    printf("Allocated %x bytes\n", arena->alloc_pos);
    
    arena_free(arena);
    if (!arena_destroy(arena)) {
        printf("Failed to destroy arena\n");
    }
}

void grad_test() {
    u32 shape[4] = {1, 1, 2, 2};
    GradTensor* gt = gradt_create(shape, 4);
    gt->tens->data[0] = 1.0;
    gt->tens->data[1] = 0.1;
    gt->tens->data[2] = -1;
    gt->tens->data[3] = 3.0;

    GradTensor* rgt = gradt_relu(gt);

    tensor_print(gt->tens, true);
    tensor_print(rgt->tens, true);
    
    gradt_free(gt);
    free(rgt);
}

void red_add_test() {
    u32 dim = 128;
    u32 shape[4] = {1, 1, dim, dim};
    Tensor* t = tensor_create(shape, 4);
    for (u32 i = 0; i < dim; i++) {
        for (u32 j = 0; j < dim; j++) {
            t->data[i * shape[3] + j] = j;
        }
    }

    double start = perf_counter_ns();
    Tensor* red = tensor_reduce_add(t, 2);
    double elapsed = perf_counter_ns() - start;
    tensor_print(t, false);
    tensor_print(red, false);
    printf("Reduction took %.3fms\n", elapsed/1000000.0);

    bool err = false;
    for (u32 i = 0; i < dim; i++) {
        f32 acc = 0.0;
        for (u32 j = 0; j < dim; j++) {
            acc += t->data[j * shape[3] + i];
        }
        if (fabsf(red->data[i] - acc) > 1e-4) {
            printf("Wrong reduced element at %d: %f - %f\n", i, red->data[i], acc);
            err = true;
        }
    }

    if (!err) {
        printf("Reduction correct\n");
    }
    
    tensor_free(t);
    tensor_free(red);
}

int main() {
    init_random();
    add_test();
    mul_test();
    mul_bt_test();
    mul_at_test();
    // arena_test();
    // grad_test();
    // red_add_test();
}
