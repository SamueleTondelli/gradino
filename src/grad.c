#include "../include/grad.h"
#include <stdbool.h>


GradTensor* create_gradt(u32* shape, usize shape_len) {
    if (shape_len > 4) {
        return NULL;
    }

    GradTensor* gt = malloc(sizeof(GradTensor));
    gt->tens = create_tensor(shape, shape_len);
    gt->grad = create_tensor(shape, shape_len);
    op_set_nop(&gt->op);
    return gt;
}

GradTensor* create_gradt_from_tens(Tensor* tens) {
    GradTensor* gt = malloc(sizeof(GradTensor));
    gt->tens = tens;
    gt->grad = create_tensor(tens->shape, 4);
    op_set_nop(&gt->op);
    return gt;
}

void free_gradt(GradTensor* gt) {
    free_tensor(gt->tens);
    free_tensor(gt->grad);
    free(gt);
}

GradTensor* relu(GradTensor* gt) {
    GradTensor* res = create_gradt(gt->tens->shape, 4);
    op_set_relu(&res->op, gt, res);
    op_fwd(&res->op);
    return res;
}

GradTensor* add(GradTensor* gt1, GradTensor* gt2) {
    Tensor* tens = add_tensor(gt1->tens, gt2->tens);
    GradTensor* gt = create_gradt_from_tens(tens);
    op_set_add(&gt->op, gt1, gt2, gt);
    return gt;
}

GradTensor* mul(GradTensor* gt1, GradTensor* gt2) {
    Tensor* tens = mul_tensor_tr(gt1->tens, gt2->tens, false, false);
    GradTensor* gt = create_gradt_from_tens(tens);
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

void backward(GradTensor* gt) {
    if (gt->tens->data_len != 1) {
        printf("Only scalar tensors allowed in backward, got %lu length\n", gt->tens->data_len);
    }
    DynArray topo = create_dynarr(10);
    DynArray visited = create_dynarr(10);

    set_tensor(gt->grad, 1.0);
    for (usize i = 0; i < topo.len; i++) {
        GradTensor* gt = (GradTensor*)topo.ptr[topo.len - i - 1];
        op_bwd(&gt->op);
    }
    
    free_dynarr(&topo);
    free_dynarr(&visited);
}
