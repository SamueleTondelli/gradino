#include "../include/grad.h"
#include <stdbool.h>

static arena_allocator* gradt_arena = NULL;

void gradt_set_arena(arena_allocator* arena) {
    gradt_arena = arena;
}

void gradt_destroy_arena() {
    arena_destroy(gradt_arena);
    gradt_arena = NULL;
}

void gradt_detach_arena() {
    gradt_arena = NULL;
}

void gradt_set_and_destroy_arena(arena_allocator* arena) {
    if (gradt_arena != NULL) {
        arena_destroy(gradt_arena);
    }
    gradt_arena = arena;
}

void gradt_free_arena() {
    arena_free(gradt_arena);
}

arena_allocator* _gradt_get_arena() {
    return gradt_arena;
}

GradTensor* gradt_create(u32* shape, usize shape_len) {
    if (shape_len > 4) {
        return NULL;
    }

    GradTensor* gt = arena_alloc(gradt_arena, sizeof(GradTensor), 1);
    gt->tens = tensor_create(shape, shape_len, gradt_arena);
    gt->grad = tensor_create(shape, shape_len, gradt_arena);
    gt->prev_grad = NULL;
    tensor_set(gt->grad, 0.0);
    gt->optimize = false;
    gt->_grad_computed = false;
    op_set_nop(&gt->op, gt);
    gt->_first_moment = NULL;
    gt->_second_moment = NULL;
    return gt;
}

GradTensor* gradt_create_from_tens(Tensor* tens) {
    GradTensor* gt = arena_alloc(gradt_arena, sizeof(GradTensor), 1);
    gt->tens = tens;
    gt->grad = tensor_create(tens->shape, 4, gradt_arena);
    gt->prev_grad = NULL;
    gt->optimize = false;
    gt->_grad_computed = false;
    tensor_set(gt->grad, 0.0);
    op_set_nop(&gt->op, gt);
    gt->_first_moment = NULL;
    gt->_second_moment = NULL;
    return gt;
}

GradTensor* gradt_create_from_labels(u32* labels, u32 n_classes, u32 n_labels) {
    u32 shape[4] = {1, 1, n_labels, n_classes};
    Tensor* t = tensor_create(shape, 4, gradt_arena);
    for (usize l = 0; l < n_labels; l++) {
        usize base = l * t->stride[2];
        for (usize i = 0; i < n_classes; i++) {
            usize idx = base + i;
            if (labels[l] == i) {
                t->data[idx] = 1.0;
            } else {
                t->data[idx] = 0.0;
            }
        }
    }
    GradTensor* gt = gradt_create_from_tens(t);
    return gt;
}

GradTensor* gradt_create_nograd(u32* shape, usize shape_len) {
    if (shape_len > 4) {
        return NULL;
    }

    GradTensor* gt = arena_alloc(gradt_arena, sizeof(GradTensor), 1);
    gt->tens = tensor_create(shape, shape_len, gradt_arena);
    gt->grad = NULL;
    gt->prev_grad = NULL;
    gt->optimize = false;
    gt->_grad_computed = false;
    op_set_nop(&gt->op, gt);
    gt->_first_moment = NULL;
    gt->_second_moment = NULL;
    return gt;
}

GradTensor** gradt_create_split_views(const GradTensor* gt, u32 split_dim) {
    if (split_dim >= 4) {
        return NULL;
    }
    Tensor** tens_views = tensor_create_split_views(gt->tens, split_dim, gradt_arena);
    Tensor** grad_views = gt->grad == NULL ? NULL : tensor_create_split_views(gt->grad, split_dim, gradt_arena);
    GradTensor** views = arena_alloc(gradt_arena, sizeof(GradTensor*), gt->tens->shape[split_dim]);
    for (u32 i = 0; i < gt->tens->shape[split_dim]; i++) {
        views[i] = arena_alloc(gradt_arena, sizeof(GradTensor), 1);
        GradTensor* v = views[i];
        v->tens = tens_views[i];
        if (grad_views != NULL) {
            v->grad = grad_views[i];
        } else {
            v->grad = NULL;
        }
        v->prev_grad = NULL;
        op_set_nop(&v->op, v);
        v->optimize = gt->optimize;
        v->_grad_computed = false;
        v->_first_moment = NULL;
        v->_second_moment = NULL;
    }
    return views;
}

void gradt_enable_optim(GradTensor* gt) {
    gt->optimize = true;
    gt->prev_grad = tensor_create(gt->tens->shape, 4, gradt_arena);
    tensor_set(gt->prev_grad, 0.0);
    gt->_first_moment = tensor_create(gt->tens->shape, 4, gradt_arena);
    tensor_set(gt->_first_moment, 0.0);
    gt->_second_moment = tensor_create(gt->tens->shape, 4, gradt_arena);
    tensor_set(gt->_second_moment, 0.0);
}

GradTensor* gradt_relu(GradTensor* gt) {
    GradTensor* res = gradt_create(gt->tens->shape, 4);
    op_set_relu(&res->op, gt, res);
    op_fwd(&res->op);
    return res;
}

GradTensor* gradt_add(GradTensor* gt1, GradTensor* gt2) {
    Tensor* tens = tensor_add(gt1->tens, gt2->tens, gradt_arena);
    GradTensor* gt = gradt_create_from_tens(tens);
    op_set_add(&gt->op, gt1, gt2, gt);
    return gt;
}

GradTensor* gradt_mul(GradTensor* gt1, GradTensor* gt2) {
    Tensor* tens = tensor_mul_tr(gt1->tens, gt2->tens, false, false, gradt_arena);
    GradTensor* gt = gradt_create_from_tens(tens);
    op_set_mul(&gt->op, gt1, gt2, gt);
    return gt;
}

static void topo_sort(GradTensor* gt, DynArray* topo, DynArray* visited) {
    if (!contains(visited, gt)) {
        push_dynarr(visited, gt);
        if (gt->op.type == Mono) {
            if (gt->op.op.mono.src != NULL) // check if it's not NOP
                topo_sort(gt->op.op.mono.src, topo, visited);
        } else {
            topo_sort(gt->op.op.bin.src1, topo, visited);
            topo_sort(gt->op.op.bin.src2, topo, visited);
        }
        push_dynarr(topo, gt);
    }
}

GradTensor* gradt_cross_entropy_loss(GradTensor* src, GradTensor* truth) {
    Tensor* t_loss = tensor_cross_entropy(src->tens, truth->tens, gradt_arena);
    GradTensor* loss = gradt_create_from_tens(t_loss);
    op_set_cse(&loss->op, src, truth, loss);
    return loss;
}

GradTensor* gradt_mean_squared_error_loss(GradTensor* src, GradTensor* truth) {
    Tensor* t_loss = tensor_mean_squared_error(src->tens, truth->tens, gradt_arena);
    GradTensor* loss = gradt_create_from_tens(t_loss);
    op_set_mse(&loss->op, src, truth, loss);
    return loss;
}

GradTensor* gradt_sigmoid(GradTensor* src) {
    Tensor* t_result = tensor_sigmoid(src->tens, gradt_arena);
    GradTensor* result = gradt_create_from_tens(t_result); 
    op_set_sigmoid(&result->op, src, result);
    return result;
}

GradTensor* gradt_mul_elemwise(GradTensor* gt1, GradTensor* gt2) {
    Tensor* t_result = tensor_mul_elemwise(gt1->tens, gt2->tens, gradt_arena);
    GradTensor* result = gradt_create_from_tens(t_result);
    op_set_mul_elemwise(&result->op, gt1, gt2, result);
    return result;
}

GradTensor* gradt_tanh(GradTensor* src) {
    Tensor* t_result = tensor_tanh(src->tens, gradt_arena);
    GradTensor* result = gradt_create_from_tens(t_result); 
    op_set_tanh(&result->op, src, result);
    return result;
}

GradTensor* gradt_concat(GradTensor* gt1, GradTensor* gt2, u32 concat_dim) {
    Tensor* c_t = tensor_concat(gt1->tens, gt2->tens, concat_dim, gradt_arena);
    GradTensor* c = gradt_create_from_tens(c_t);
    op_set_concat(&c->op, gt1, gt2, c, concat_dim);
    return c;
}

f32 gradt_compute_accuracy(GradTensor* src, GradTensor* truth) {
    Tensor* s = src->tens;
    Tensor* t = truth->tens;
    for (u32 i = 0; i < 4; i++) {
        if (s->shape[i] != t->shape[i]) {
            printf("gradt_compute_accuracy: shape mismatch at dim %u: %u vs %u\n", i, s->shape[i], t->shape[i]);
            exit(1);
        }
    }
    u32 n_samples = s->shape[2];
    u32 n_classes = s->shape[3];
    u32 correct = 0;
    for (u32 i = 0; i < n_samples; i++) {
        u32 pred_class = 0;
        u32 true_class = 0;
        f32 pred_max = s->data[i * n_classes];
        f32 true_max = t->data[i * n_classes];
        for (u32 c = 1; c < n_classes; c++) {
            if (s->data[i * n_classes + c] > pred_max) {
                pred_max = s->data[i * n_classes + c];
                pred_class = c;
            }
            if (t->data[i * n_classes + c] > true_max) {
                true_max = t->data[i * n_classes + c];
                true_class = c;
            }
        }
        if (pred_class == true_class) correct++;
    }
    return (f32)correct / (f32)n_samples;
}

void gradt_backward(GradTensor* gt, Optimizer optim, void* optim_config) {
    if (gt->tens->data_len != 1) {
        printf("Only scalar tensors allowed in backward, got %lu length\n", gt->tens->data_len);
        return;
    }
    tensor_set(gt->grad, 1.0);
    
    DynArray topo = create_dynarr(10);
    DynArray visited = create_dynarr(10);
    topo_sort(gt, &topo, &visited);
    for (usize i = 0; i < topo.len - 1; i++) {
        GradTensor* gti = (GradTensor*)topo.ptr[i];
        if (gti->grad == NULL || gti->prev_grad == NULL) {
            continue;
        }
        Tensor* temp = gti->grad;
        gti->grad = gti->prev_grad;
        gti->prev_grad = temp;
        tensor_set(gti->grad, 0.0);
        gti->_grad_computed = false;
    }
    
    for (usize i = 0; i < topo.len; i++) {
        GradTensor* gti = (GradTensor*)topo.ptr[topo.len - i - 1];
        op_bwd(&gti->op);
    }

    for (usize i = 0; i < topo.len; i++) {
        GradTensor* gti = (GradTensor*)topo.ptr[topo.len - i - 1];
        if (gti->optimize) {
            optim(gti, optim_config);
        }
    }
    
    free_dynarr(&topo);
    free_dynarr(&visited);
}
