#include "../include/optim.h"

void optim_sgd(GradTensor* gt, void* sgd_config) {
    SGDConfig* config = (SGDConfig*)sgd_config;
    _tensor_kernel_sub_scaled(gt->tens, gt->grad, config->lr, gt->tens);
}

SGDConfig optim_sgd_get_config(f32 lr) {
    SGDConfig c = { .lr = lr };
    return c;
}
