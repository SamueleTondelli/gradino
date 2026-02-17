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
    tensor_randomize(in->tens, 0.0, 1.0);
    u32 true_labels[4] = {1, 4, 0, 12};    
    printf("    Creating label tensor\n");
    GradTensor* labels = gradt_create_from_labels(true_labels, 16, 4);
    printf("    Creating linear layer\n");
    LinearLayer lin = nn_linear_create(8, 16);
    for (i32 i = 0; i < 5; i++) {
        printf("        Epoch %d, ", i);
        GradTensor* preact = nn_linear_forward(&lin, in);
        GradTensor* act = nn_relu(preact);
        GradTensor* loss = nn_cross_enropy_loss(act, labels);
        printf("Loss: %f\n", loss->tens->data[0]);

        gradt_backward(loss, optim_sgd_momentum, &sgd_momentum_config);
    }

    printf("    Destroying gradt arena\n");
    gradt_destroy_arena();
}

void test_xor(f32 lr, u32 hidden_size, u32 epochs) {
    printf("test_xor with lr %f, hidden_size %u, for %u epochs\n", lr, hidden_size, epochs);
    arena_allocator* permanent_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(permanent_arena);

    AdamConfig adam_config = optim_adam_get_config(lr);
    
    u32 in_shape[4] = {1, 1, 4, 2};
    GradTensor* in = gradt_create_nograd(in_shape, 4);
    in->tens->data[0] = 0.0;
    in->tens->data[1] = 0.0;
    in->tens->data[2] = 1.0;
    in->tens->data[3] = 0.0;
    in->tens->data[4] = 0.0;
    in->tens->data[5] = 1.0;
    in->tens->data[6] = 1.0;
    in->tens->data[7] = 1.0;

    u32 truth_shape[4] = {1, 1, 4, 1};
    GradTensor* truth = gradt_create_nograd(truth_shape, 4);
    truth->tens->data[0] = 0.0;
    truth->tens->data[1] = 1.0;
    truth->tens->data[2] = 1.0;
    truth->tens->data[3] = 0.0;

    LinearLayer hidden = nn_linear_create(2, hidden_size);
    LinearLayer out_layer = nn_linear_create(hidden_size, 1);

    arena_allocator* epoch_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(epoch_arena);
    for (u32 i = 0; i < epochs; i++) {
        GradTensor* x = nn_linear_forward(&hidden, in);
        x = nn_relu(x);
        x = nn_linear_forward(&out_layer, x);
        GradTensor* loss = nn_mean_squared_error_loss(x, truth);
        gradt_backward(loss, optim_adam, &adam_config);
        optim_adam_step(&adam_config);
        printf("    Epoch %u, Loss %f\n", i, loss->tens->data[0]);
        gradt_free_arena();
    }
    GradTensor* x = nn_linear_forward(&hidden, in);
    x = nn_relu(x);
    x = nn_linear_forward(&out_layer, x);
    printf("    0 ^ 0 = %f\n    1 ^ 0 = %f\n    0 ^ 1 = %f\n    1 ^ 1 = %f\n", x->tens->data[0], x->tens->data[1], x->tens->data[2], x->tens->data[3]);

    printf("    Destroying gradt arena\n");
    gradt_destroy_arena();
    arena_destroy(permanent_arena);
}

void test_bwd_perf(f32 lr, u32 hidden_size, u32 bs, u32 io_dim, u32 n_batches, u32 epochs) {
    printf("test_bwd_perf with lr %f, hidden_size %u, batch size %u, io dim %u, %u batches, for %u epochs\n",
           lr, hidden_size, bs, io_dim, n_batches, epochs);
    arena_allocator* permanent_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(permanent_arena);

    // SGDMomentumConfig optim_conf = optim_sgd_momentum_get_config(lr, 0.9);
    AdamConfig adam_conf = optim_adam_get_config(lr);

    u32 inout_shape[4] = {1, 1, bs, io_dim};
    GradTensor** inputs = malloc(n_batches * sizeof(GradTensor*));
    GradTensor** truths = malloc(n_batches * sizeof(GradTensor*));
    for (u32 b = 0; b < n_batches; b++) {
        inputs[b] = gradt_create_nograd(inout_shape, 4);
        tensor_randomize_gaussian(inputs[b]->tens, 0.0, 1.0);
        truths[b] = gradt_create_nograd(inout_shape, 4);
        tensor_randomize_gaussian(truths[b]->tens, 0.0, 1.0);
    }

    LinearLayer l1 = nn_linear_create(io_dim, hidden_size);
    LinearLayer l2 = nn_linear_create(hidden_size, hidden_size);
    LinearLayer l3 = nn_linear_create(hidden_size, hidden_size);
    LinearLayer l4 = nn_linear_create(hidden_size, io_dim);

    arena_allocator* epoch_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(epoch_arena);
    double total_epoch_ms = 0.0;
    for (u32 i = 0; i < epochs; i++) {
        double epoch_fwd_ms = 0.0, epoch_bwd_ms = 0.0, epoch_free_ms = 0.0;
        u64 epoch_start = perf_counter_ns();
        for (u32 b = 0; b < n_batches; b++) {
            u64 batch_start = perf_counter_ns();
            GradTensor* x = nn_linear_forward(&l1, inputs[b]);
            x = nn_relu(x);
            x = nn_linear_forward(&l2, x);
            x = nn_sigmoid(x);
            x = nn_linear_forward(&l3, x);
            x = nn_tanh(x);
            x = nn_linear_forward(&l4, x);
            GradTensor* loss = nn_mean_squared_error_loss(x, truths[b]);
            u64 fwd_end = perf_counter_ns();
            // gradt_backward(loss, optim_sgd_momentum, &optim_conf);
            gradt_backward(loss, optim_adam, &adam_conf);
            optim_adam_step(&adam_conf);
            u64 bwd_end = perf_counter_ns();
            arena_free(epoch_arena);
            u64 free_end = perf_counter_ns();

            epoch_fwd_ms  += (fwd_end - batch_start) / 1e6;
            epoch_bwd_ms  += (bwd_end - fwd_end) / 1e6;
            epoch_free_ms += (free_end - bwd_end) / 1e6;
        }
        double epoch_ms = (perf_counter_ns() - epoch_start) / 1e6;
        total_epoch_ms += epoch_ms;
        printf("    Epoch %u: fwd %.3f ms, bwd %.3f ms, free %.3f ms, total %.3f ms, avg batch %.3f ms\n",
               i, epoch_fwd_ms, epoch_bwd_ms, epoch_free_ms, epoch_ms, epoch_ms / n_batches);
    }
    printf("    Average epoch time: %.3f ms\n", total_epoch_ms / epochs);

    free(inputs);
    free(truths);
    gradt_destroy_arena();
    arena_destroy(permanent_arena);
}
