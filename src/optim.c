#include "../include/optim.h"
#include "../include/grad.h"
#include <math.h>

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

void optim_adam(GradTensor* gt, void* adam_config) {
    AdamConfig* config = (AdamConfig*)adam_config;
    arena_allocator* arena = _gradt_get_arena();

    _tensor_kernel_mul_scalar(gt->_first_moment, config->beta_1, gt->_first_moment);
    Tensor* m1_update = tensor_mul_scalar(gt->grad, 1 - config->beta_1, arena);
    _tensor_kernel_add(gt->_first_moment, m1_update, gt->_first_moment);

    _tensor_kernel_mul_scalar(gt->_second_moment, config->beta_2, gt->_second_moment);
    Tensor* m2_update = tensor_mul_elemwise(gt->grad, gt->grad, arena);
    _tensor_kernel_mul_scalar(m2_update, 1 - config->beta_2, m2_update);
    _tensor_kernel_add(gt->_second_moment, m2_update, gt->_second_moment);

    f32 m1_scale = 1 / (1 - powf(config->beta_1, config->_t));
    f32 m2_scale = 1 / (1 - powf(config->beta_2, config->_t));

    Tensor* m1_scaled = tensor_mul_scalar(gt->_first_moment, m1_scale, arena);
    Tensor* m2_scaled = tensor_mul_scalar(gt->_second_moment, m2_scale, arena);

    _tensor_kernel_adam_update(gt->tens, m1_scaled, m2_scaled, config->epsilon, config->lr);
}

AdamConfig optim_adam_get_config(f32 lr) {
    AdamConfig c = {
        .lr = lr,
        .beta_1 = 0.9,
        .beta_2 = 0.999,
        .epsilon = 1e-8,
        ._t = 1.0
    };
    return c;
}

void optim_adam_step(AdamConfig* config) {
    config->_t += 1.0;
}
