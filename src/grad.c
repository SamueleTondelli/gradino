#include "../include/grad.h"


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
