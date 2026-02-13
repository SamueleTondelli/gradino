#include "../include/nn.h"

LinearLayer nn_linear_create(u32 in, u32 out) {
    u32 w_shape[4] = {1, 1, in, out};
    u32 b_shape[4] = {1, 1, 1, out};
    LinearLayer l = {
        .w = gradt_create(w_shape, 4),
        .b = gradt_create(b_shape, 4),
        ._proj = NULL
    };
    // xavier init?
    return l;
}

GradTensor* nn_linear_forward(LinearLayer* layer, GradTensor* in) {
    layer->_proj = gradt_mul(in, layer->w);
    // not weights, shouldnt be optimized
    layer->_proj->optimize = false;
    return gradt_add(layer->_proj, layer->b);
}

GradTensor* nn_relu(GradTensor* gt) {
    return gradt_relu(gt);
}

GradTensor* nn_cross_enropy_loss(GradTensor* src, GradTensor* truth) {
    return gradt_cross_entropy_loss(src, truth);
}
