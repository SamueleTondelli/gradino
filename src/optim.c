#include "../include/optim.h"
#include "../include/grad.h"

void optim_sgd(GradTensor* gt, void* sgd_config) {
    SGDConfig* config = (SGDConfig*)sgd_config;
    _tensor_kernel_sub_scaled(gt->tens, gt->grad, config->lr, gt->tens);
}

SGDConfig optim_sgd_get_config(f32 lr) {
    SGDConfig c = { .lr = lr };
    return c;
}

void optim_sgd_momentum(GradTensor* gt, void* sgd_momentum_config) {
    SGDMomentumConfig* config = (SGDMomentumConfig*)sgd_momentum_config;
    // fuse?
    Tensor* update = tensor_add_scaled(gt->grad, gt->prev_grad, config->mu, _gradt_get_arena());
    _tensor_kernel_sub_scaled(gt->tens, update, config->lr, gt->tens);
}

SGDMomentumConfig optim_sgd_momentum_get_config(f32 lr, f32 mu) {
    SGDMomentumConfig c = { .lr = lr, .mu = mu };
    return c;
}
