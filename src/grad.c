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
