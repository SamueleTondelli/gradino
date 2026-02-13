#ifndef OPTIM_H
#define OPTIM_H

#include "grad.h"

typedef struct {
    f32 lr;
} SGDConfig;

void optim_sgd(GradTensor* gt, void* sgd_config);
SGDConfig optim_sgd_get_config(f32 lr);

typedef struct {
    f32 lr;
    f32 mu;
} SGDMomentumConfig;

void optim_sgd_momentum(GradTensor* gt, void* sgd_momentum_config);
SGDMomentumConfig optim_sgd_momentum_get_config(f32 lr, f32 mu);

#endif
