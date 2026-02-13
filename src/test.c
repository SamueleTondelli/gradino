#include "../include/test.h"
#include "../include/arena.h"
#include "../include/tensor.h"
#include "../include/grad.h"
#include "../include/utils.h"
#include "../include/optim.h"
#include "../include/nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void ref_matmul(const f32* a, const f32* b, f32* res,
                       u32 m, u32 k, u32 n, bool at, bool bt) {
    for (u32 i = 0; i < m; i++) {
        for (u32 j = 0; j < n; j++) {
            f32 el = 0.0f;
            for (u32 l = 0; l < k; l++) {
                f32 av = at ? a[l * m + i] : a[i * k + l];
                f32 bv = bt ? b[j * k + l] : b[l * n + j];
                el += av * bv;
            }
            res[i * n + j] = el;
        }
    }
}

static bool verify_data(const f32* got, const f32* expect, u32 rows, u32 cols, f32 tol) {
    for (u32 i = 0; i < rows; i++) {
        for (u32 j = 0; j < cols; j++) {
            u32 idx = i * cols + j;
            if (fabsf(got[idx] - expect[idx]) > tol) {
                printf("  FAIL at (%u, %u): got %f, expected %f\n", i, j, got[idx], expect[idx]);
                return false;
            }
        }
    }
    return true;
}

void test_add(u32 rows, u32 cols) {
    printf("test_add [%u x %u]\n", rows, cols);

    arena_allocator* arena = arena_create(GiB(1), MiB(1), 8);

    u32 shape[] = {1, 1, rows, cols};
    Tensor* a = tensor_create(shape, 4, arena);
    Tensor* b = tensor_create(shape, 4, arena);
    tensor_randomize(a, 0.0f, 1.0f);
    tensor_randomize(b, 0.0f, 1.0f);

    double start = perf_counter_ns();
    Tensor* c = tensor_add(a, b, arena);
    double elapsed_ms = (perf_counter_ns() - start) / 1e6;

    bool ok = true;
    for (usize i = 0; i < c->data_len; i++) {
        if (fabsf(c->data[i] - (a->data[i] + b->data[i])) > 1e-6f) {
            printf("  FAIL at %zu: got %f, expected %f\n", i, c->data[i], a->data[i] + b->data[i]);
            ok = false;
            break;
        }
    }

    printf("  %s  %.3f ms\n", ok ? "PASS" : "FAIL", elapsed_ms);

    arena_destroy(arena);
}

static void run_mul_variant(const char* label, u32 m, u32 k, u32 n, bool at, bool bt,
                            arena_allocator* arena) {
    u32 a_rows = at ? k : m;
    u32 a_cols = at ? m : k;
    u32 b_rows = bt ? n : k;
    u32 b_cols = bt ? k : n;

    u32 a_shape[] = {1, 1, a_rows, a_cols};
    u32 b_shape[] = {1, 1, b_rows, b_cols};

    Tensor* a = tensor_create(a_shape, 4, arena);
    Tensor* b = tensor_create(b_shape, 4, arena);
    tensor_randomize(a, 0.0f, 1.0f);
    tensor_randomize(b, 0.0f, 1.0f);

    double start = perf_counter_ns();
    Tensor* c = tensor_mul_tr(a, b, at, bt, arena);
    double elapsed_ms = (perf_counter_ns() - start) / 1e6;

    f32* ref = malloc(m * n * sizeof(f32));
    ref_matmul(a->data, b->data, ref, m, k, n, at, bt);
    bool ok = verify_data(c->data, ref, m, n, 1e-3f);

    printf("  %-12s %s  %.3f ms\n", label, ok ? "PASS" : "FAIL", elapsed_ms);

    free(ref);
    arena_destroy(arena);
}

void test_mul(u32 m, u32 k, u32 n) {
    printf("test_mul [%u x %u] * [%u x %u]\n", m, k, k, n);
    run_mul_variant("A*B",  m, k, n, false, false, arena_create(GiB(1), MiB(1), 8));
    run_mul_variant("At*B", m, k, n, true,  false, arena_create(GiB(1), MiB(1), 8));
    run_mul_variant("A*Bt", m, k, n, false, true,  arena_create(GiB(1), MiB(1), 8));
}

void test_reduce_add(u32 rows, u32 cols, u32 dim) {
    printf("test_reduce_add [%u x %u] dim=%u\n", rows, cols, dim);

    arena_allocator* arena = arena_create(GiB(1), MiB(1), 8);

    u32 shape[] = {1, 1, rows, cols};
    Tensor* t = tensor_create(shape, 4, arena);
    tensor_randomize(t, 0.0f, 10.0f);

    double start = perf_counter_ns();
    Tensor* red = tensor_reduce_add(t, dim, arena);
    double elapsed_ms = (perf_counter_ns() - start) / 1e6;

    u32 outer = (dim == 2) ? cols : rows;
    u32 inner = (dim == 2) ? rows : cols;
    bool ok = true;
    for (u32 i = 0; i < outer && ok; i++) {
        f32 acc = 0.0f;
        for (u32 j = 0; j < inner; j++) {
            usize idx = (dim == 2) ? (j * cols + i) : (i * cols + j);
            acc += t->data[idx];
        }
        if (fabsf(red->data[i] - acc) > 1e-2f) {
            printf("  FAIL at %u: got %f, expected %f\n", i, red->data[i], acc);
            ok = false;
        }
    }

    printf("  %s  %.3f ms\n", ok ? "PASS" : "FAIL", elapsed_ms);

    arena_destroy(arena);
}

void test_arena(usize reserve, usize commit, usize alloc_size, u32 n_allocs) {
    printf("test_arena  reserve=%zuMiB commit=%zuKiB alloc=%zuKiB x%u\n",
           reserve >> 20, commit >> 10, alloc_size >> 10, n_allocs);

    arena_allocator* arena = arena_create(reserve, commit, 8);
    if (!arena) {
        printf("  FAIL: arena_create returned NULL\n");
        return;
    }

    bool ok = true;
    for (u32 i = 0; i < n_allocs; i++) {
        u8* mem = arena_alloc(arena, alloc_size, 1);
        if (!mem) {
            printf("  FAIL: allocation %u returned NULL\n", i);
            ok = false;
            break;
        }
        mem[0] = 1;
    }

    printf("  allocated %zu bytes total\n", arena->alloc_pos);
    arena_free(arena);

    if (!arena_destroy(arena)) {
        printf("  FAIL: arena_destroy failed\n");
        ok = false;
    }

    printf("  %s\n", ok ? "PASS" : "FAIL");
}

void test_grad_relu() {
    printf("test_grad_relu [2 x 2]\n");

    arena_allocator* arena = arena_create(GiB(1), MiB(1), 8);
    if (!arena) {
        printf("  FAIL: arena_create returned NULL\n");
        return;
    }
    gradt_set_arena(arena);

    u32 shape[] = {1, 1, 2, 2};
    GradTensor* gt = gradt_create(shape, 4);
    gt->tens->data[0] =  1.0f;
    gt->tens->data[1] =  0.1f;
    gt->tens->data[2] = -1.0f;
    gt->tens->data[3] =  3.0f;

    GradTensor* rgt = gradt_relu(gt);

    bool ok = true;
    f32 expect[] = {1.0f, 0.1f, 0.0f, 3.0f};
    for (int i = 0; i < 4; i++) {
        if (fabsf(rgt->tens->data[i] - expect[i]) > 1e-6f) {
            printf("  FAIL at %d: got %f, expected %f\n", i, rgt->tens->data[i], expect[i]);
            ok = false;
        }
    }

    printf("  %s\n", ok ? "PASS" : "FAIL");

    gradt_destroy_arena();
}

void test_grad_bwd() {
    printf("test_grad_bwd\n");

    arena_allocator* arena = arena_create(GiB(1), MiB(1), 8);
    if (!arena) {
        printf("  FAIL: arena_create returned NULL\n");
        return;
    }

    gradt_set_arena(arena);

    SGDMomentumConfig sgd_momentum_config = optim_sgd_momentum_get_config(1e-3, 0.9);
        
    u32 in_shape[4] = {1, 1, 4, 8};
    printf("    Creating input batch\n");
    GradTensor* in = gradt_create_nograd(in_shape, 4);
    u32 true_labels[4] = {1, 4, 0, 12};    
    printf("    Creating label tensor\n");
    GradTensor* labels = gradt_create_from_labels(true_labels, 16, 4, false);

    printf("    Creating linear layer\n");
    LinearLayer lin = nn_linear_create(8, 16);
    printf("    linear forward\n");
    GradTensor* preact = nn_linear_forward(&lin, in);
    printf("    relu(preact)\n");
    GradTensor* act = gradt_relu(preact);
    printf("    Computing loss\n");
    GradTensor* loss = gradt_cross_entropy_loss(act, labels);

    printf("    Backward pass\n");
    gradt_backward(loss, optim_sgd_momentum, &sgd_momentum_config);

    printf("    Destroying gradt arena\n");
    gradt_destroy_arena();
}
