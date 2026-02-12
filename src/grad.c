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
    op_set_nop(&gt->op);
    return gt;
}

GradTensor* gradt_create_from_tens(Tensor* tens) {
    GradTensor* gt = arena_alloc(gradt_arena, sizeof(GradTensor), 1);
    gt->tens = tens;
    gt->grad = tensor_create(tens->shape, 4, gradt_arena);
    op_set_nop(&gt->op);
    return gt;
}

GradTensor* gradt_create_from_labels(u32* labels, u32 n_classes, u32 n_labels) {
    u32 shape[4] = {1, 1, n_labels, n_classes};
    Tensor* t = tensor_create(shape, 4, gradt_arena);
    for (usize l = 0; l < n_labels; l++) {
        usize base = t->stride[2];
        for (usize i = 0; i < n_labels; i++) {
            usize idx = base + i;
            if (labels[l] == i) {
                t->data[idx] = 1.0;
            } else {
                t->data[idx] = 0.0;
            }
        }
    }
    return gradt_create_from_tens(t);
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

void gradt_backward(GradTensor* gt) {
    if (gt->tens->data_len != 1) {
        printf("Only scalar tensors allowed in backward, got %lu length\n", gt->tens->data_len);
    }
    tensor_set(gt->grad, 1.0);
    
    DynArray topo = create_dynarr(10);
    DynArray visited = create_dynarr(10);
    topo_sort(gt, &topo, &visited);
    for (usize i = 0; i < topo.len; i++) {
        GradTensor* gt = (GradTensor*)topo.ptr[topo.len - i - 1];
        op_bwd(&gt->op);
    }
    
    free_dynarr(&topo);
    free_dynarr(&visited);
}
