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

void test_xor(f32 lr, f32 weight_decay, u32 hidden_size, u32 epochs) {
    printf("test_xor with lr %f, weight decay %f, hidden_size %u, for %u epochs\n", lr, weight_decay, hidden_size, epochs);
    arena_allocator* permanent_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(permanent_arena);

    AdamWConfig adamw_config = optim_adamw_get_config(lr, weight_decay);
    
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
        gradt_backward(loss, optim_adamw, &adamw_config);
        optim_adamw_step(&adamw_config);
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
    // AdamConfig adam_conf = optim_adam_get_config(lr);
    AdamWConfig adamw_conf = optim_adamw_get_config(lr, 1e-4);

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
            // gradt_backward(loss, optim_adam, &adam_conf);
            // optim_adam_step(&adam_conf);
            gradt_backward(loss, optim_adamw, &adamw_conf);
            optim_adamw_step(&adamw_conf);
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

void test_split_views() {
    printf("test_split_views\n");
    arena_allocator* arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(arena);

    u32 shape[4] = {1, 4, 2, 3};
    GradTensor* gt = gradt_create(shape, 4);

    for (usize i = 0; i < gt->tens->data_len; i++) {
        gt->tens->data[i] = (f32)i;
        gt->grad->data[i] = (f32)(i + 100);
    }

    GradTensor** views = gradt_create_split_views(gt, 1);
    if (views == NULL) {
        printf("    gradt_create_split_views returned NULL\n");
        goto cleanup;
    }

    for (u32 i = 0; i < 4; i++) {
        GradTensor* v = views[i];

        if (v->tens->data_len != 6) {
            printf("    views[%u] tens has wrong data_len %lu\n", i, v->tens->data_len);
            goto cleanup;
        }
        if (v->grad->data_len != 6) {
            printf("    views[%u] grad has wrong data_len %lu\n", i, v->grad->data_len);
            goto cleanup;
        }

        if (v->tens->shape[0] != 1 || v->tens->shape[1] != 1 ||
            v->tens->shape[2] != gt->tens->shape[2] || v->tens->shape[3] != gt->tens->shape[3]) {
            printf("    views[%u] tens has incorrect shape (%u, %u, %u, %u)\n",
                   i, v->tens->shape[0], v->tens->shape[1], v->tens->shape[2], v->tens->shape[3]);
            goto cleanup;
        }

        if (v->grad->shape[0] != 1 || v->grad->shape[1] != 1 ||
            v->grad->shape[2] != gt->grad->shape[2] || v->grad->shape[3] != gt->grad->shape[3]) {
            printf("    views[%u] grad has incorrect shape (%u, %u, %u, %u)\n",
                   i, v->grad->shape[0], v->grad->shape[1], v->grad->shape[2], v->grad->shape[3]);
            goto cleanup;
        }

        if (v->tens->stride[0] != 0 || v->tens->stride[1] != 0 ||
            v->tens->stride[2] != gt->tens->stride[2] || v->tens->stride[3] != gt->tens->stride[3]) {
            printf("    views[%u] tens has incorrect stride (%u, %u, %u, %u)\n",
                   i, v->tens->stride[0], v->tens->stride[1], v->tens->stride[2], v->tens->stride[3]);
            goto cleanup;
        }

        if (v->grad->stride[0] != 0 || v->grad->stride[1] != 0 ||
            v->grad->stride[2] != gt->grad->stride[2] || v->grad->stride[3] != gt->grad->stride[3]) {
            printf("    views[%u] grad has incorrect stride (%u, %u, %u, %u)\n",
                   i, v->grad->stride[0], v->grad->stride[1], v->grad->stride[2], v->grad->stride[3]);
            goto cleanup;
        }

        for (u32 j = 0; j < v->tens->data_len; j++) {
            usize t_idx = i * 6 + j;
            if (v->tens->data[j] != gt->tens->data[t_idx]) {
                printf("    views[%u] tens[%u] != gt->tens[%lu], %f != %f\n",
                       i, j, t_idx, v->tens->data[j], gt->tens->data[t_idx]);
                goto cleanup;
            }
            if (v->grad->data[j] != gt->grad->data[t_idx]) {
                printf("    views[%u] grad[%u] != gt->grad[%lu], %f != %f\n",
                       i, j, t_idx, v->grad->data[j], gt->grad->data[t_idx]);
                goto cleanup;
            }
        }
    }

    printf("    test_split_views PASSED\n");

cleanup:
    gradt_destroy_arena();
}

static usize idx4(const u32* stride, u32 d0, u32 d1, u32 d2, u32 d3) {
    return d0 * stride[0] + d1 * stride[1] + d2 * stride[2] + d3 * stride[3];
}

static bool test_concat_dim(u32* a_shape, u32* b_shape, u32 concat_dim) {
    GradTensor* a = gradt_create(a_shape, 4);
    GradTensor* b = gradt_create(b_shape, 4);

    for (u32 i = 0; i < a->tens->data_len; i++) {
        a->tens->data[i] = (f32)i;
        a->grad->data[i] = 0.0f;
    }
    for (u32 i = 0; i < b->tens->data_len; i++) {
        b->tens->data[i] = (f32)(i * 10 + i);
        b->grad->data[i] = 0.0f;
    }

    GradTensor* c = gradt_concat(a, b, concat_dim);
    if (c == NULL) {
        printf("    concat(%u) returned NULL\n", concat_dim);
        return false;
    }

    u32 exp_shape[4];
    for (u32 i = 0; i < 4; i++) {
        if (i < concat_dim)
            exp_shape[i] = a_shape[i] > b_shape[i] ? a_shape[i] : b_shape[i];
        else if (i == concat_dim)
            exp_shape[i] = a_shape[i] + b_shape[i];
        else
            exp_shape[i] = a_shape[i];
    }
    for (u32 i = 0; i < 4; i++) {
        if (c->tens->shape[i] != exp_shape[i]) {
            printf("    concat(%u) bad shape (%u,%u,%u,%u) expected (%u,%u,%u,%u)\n", concat_dim,
                   c->tens->shape[0], c->tens->shape[1], c->tens->shape[2], c->tens->shape[3],
                   exp_shape[0], exp_shape[1], exp_shape[2], exp_shape[3]);
            return false;
        }
    }

    bool ok = true;

    // verify forward: iterate over all elements of the result
    for (u32 d0 = 0; d0 < c->tens->shape[0] && ok; d0++) {
        for (u32 d1 = 0; d1 < c->tens->shape[1] && ok; d1++) {
            for (u32 d2 = 0; d2 < c->tens->shape[2] && ok; d2++) {
                for (u32 d3 = 0; d3 < c->tens->shape[3] && ok; d3++) {
                    u32 coord[4] = {d0, d1, d2, d3};
                    usize c_idx = idx4(c->tens->stride, d0, d1, d2, d3);

                    if (coord[concat_dim] < a_shape[concat_dim]) {
                        usize a_idx = idx4(a->tens->stride, d0, d1, d2, d3);
                        if (c->tens->data[c_idx] != a->tens->data[a_idx]) {
                            printf("    concat(%u) FAIL fwd a at (%u,%u,%u,%u): got %f, expected %f\n",
                                   concat_dim, d0, d1, d2, d3, c->tens->data[c_idx], a->tens->data[a_idx]);
                            ok = false;
                        }
                    } else {
                        u32 b_coord[4] = {d0, d1, d2, d3};
                        b_coord[concat_dim] -= a_shape[concat_dim];
                        usize b_idx = idx4(b->tens->stride, b_coord[0], b_coord[1], b_coord[2], b_coord[3]);
                        if (c->tens->data[c_idx] != b->tens->data[b_idx]) {
                            printf("    concat(%u) FAIL fwd b at (%u,%u,%u,%u): got %f, expected %f\n",
                                   concat_dim, d0, d1, d2, d3, c->tens->data[c_idx], b->tens->data[b_idx]);
                            ok = false;
                        }
                    }
                }
            }
        }
    }

    printf("    concat(%u) fwd %s\n", concat_dim, ok ? "PASS" : "FAIL");

    // backward: set result grad to known values, run op_bwd, check a/b grads
    for (usize i = 0; i < c->grad->data_len; i++) {
        c->grad->data[i] = (f32)(i + 500);
    }

    op_bwd(&c->op);

    // build expected grads by accumulating over broadcast dims
    f32* a_expect = calloc(a->grad->data_len, sizeof(f32));
    f32* b_expect = calloc(b->grad->data_len, sizeof(f32));

    for (u32 d0 = 0; d0 < c->tens->shape[0]; d0++) {
        for (u32 d1 = 0; d1 < c->tens->shape[1]; d1++) {
            for (u32 d2 = 0; d2 < c->tens->shape[2]; d2++) {
                for (u32 d3 = 0; d3 < c->tens->shape[3]; d3++) {
                    u32 coord[4] = {d0, d1, d2, d3};
                    usize c_idx = idx4(c->grad->stride, d0, d1, d2, d3);

                    if (coord[concat_dim] < a_shape[concat_dim]) {
                        u32 ac[4] = {d0, d1, d2, d3};
                        for (u32 k = 0; k < 4; k++) {
                            if (a_shape[k] == 1) ac[k] = 0;
                        }
                        usize a_idx = idx4(a->grad->stride, ac[0], ac[1], ac[2], ac[3]);
                        a_expect[a_idx] += c->grad->data[c_idx];
                    } else {
                        u32 bc[4] = {d0, d1, d2, d3};
                        bc[concat_dim] -= a_shape[concat_dim];
                        for (u32 k = 0; k < 4; k++) {
                            if (b_shape[k] == 1) bc[k] = 0;
                        }
                        usize b_idx = idx4(b->grad->stride, bc[0], bc[1], bc[2], bc[3]);
                        b_expect[b_idx] += c->grad->data[c_idx];
                    }
                }
            }
        }
    }

    bool bwd_ok = true;
    for (usize i = 0; i < a->grad->data_len && bwd_ok; i++) {
        if (fabsf(a->grad->data[i] - a_expect[i]) > 1e-3f) {
            printf("    concat(%u) FAIL bwd a grad at flat %zu: got %f, expected %f\n",
                   concat_dim, i, a->grad->data[i], a_expect[i]);
            bwd_ok = false;
        }
    }
    for (usize i = 0; i < b->grad->data_len && bwd_ok; i++) {
        if (fabsf(b->grad->data[i] - b_expect[i]) > 1e-3f) {
            printf("    concat(%u) FAIL bwd b grad at flat %zu: got %f, expected %f\n",
                   concat_dim, i, b->grad->data[i], b_expect[i]);
            bwd_ok = false;
        }
    }

    free(a_expect);
    free(b_expect);

    printf("    concat(%u) bwd %s\n", concat_dim, bwd_ok ? "PASS" : "FAIL");
    return ok && bwd_ok;
}

void test_concat() {
    printf("test_concat\n");
    arena_allocator* arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(arena);

    // no broadcasting
    u32 a3[] = {1, 2, 3, 4}, b3[] = {1, 2, 3, 5};
    test_concat_dim(a3, b3, 3);

    u32 a2[] = {1, 2, 1, 4}, b2[] = {1, 2, 3, 4};
    test_concat_dim(a2, b2, 2);

    // broadcasting on dims before concat_dim
    u32 a1[] = {1, 2, 1, 4}, b1[] = {1, 1, 3, 4};
    test_concat_dim(a1, b1, 3);

    u32 a0[] = {2, 1, 3, 4}, b0[] = {1, 2, 3, 4};
    test_concat_dim(a0, b0, 2);

    gradt_destroy_arena();
}

void test_lstm(u32 in_size, u32 hidden_size, u32 bs, u32 seq_len, u32 n_batches, u32 epochs) {
    printf("test_lstm with in_size %u, hidden_size %u, batch size %u, seq_len %u, %u batches, for %u epochs\n",
           in_size, hidden_size, bs, seq_len, n_batches, epochs);
    arena_allocator* permanent_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(permanent_arena);

    AdamWConfig config = optim_adamw_get_config(1e-3, 1e-5);

    u32 input_shape[4] = {1, seq_len, bs, in_size};
    u32 truth_shape[4] = {1, 1, bs, hidden_size};
    GradTensor** inputs = malloc(n_batches * sizeof(GradTensor*));
    GradTensor** truths = malloc(n_batches * sizeof(GradTensor*));
    for (u32 b = 0; b < n_batches; b++) {
        inputs[b] = gradt_create_nograd(input_shape, 4);
        tensor_randomize(inputs[b]->tens, 0.0, 1.0);
        truths[b] = gradt_create_nograd(truth_shape, 4);
        tensor_randomize(truths[b]->tens, 0.0, 1.0);
    }

    LSTM lstm = nn_lstm_init(in_size, hidden_size);

    arena_allocator* epoch_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(epoch_arena);
    double total_epoch_ms = 0.0;
    for (u32 i = 0; i < epochs; i++) {
        double epoch_fwd_ms = 0.0, epoch_bwd_ms = 0.0, epoch_free_ms = 0.0;
        u64 epoch_start = perf_counter_ns();
        for (u32 b = 0; b < n_batches; b++) {
            u64 batch_start = perf_counter_ns();
            GradTensor* x = nn_lstm_forward(&lstm, inputs[b]);
            x = nn_cross_enropy_loss(x, truths[b]);
            u64 fwd_end = perf_counter_ns();
            gradt_backward(x, optim_adamw, &config);
            optim_adamw_step(&config);
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
