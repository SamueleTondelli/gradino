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

typedef struct {
    f32 lr;
    f32 beta_1;
    f32 beta_2;
    f32 epsilon;
    f32 _t;
} AdamConfig;

void optim_adam(GradTensor* gt, void* adam_config);
AdamConfig optim_adam_get_config(f32 lr);
void optim_adam_step(AdamConfig* config);

typedef struct {
    AdamConfig adam_config;
    f32 weight_decay;
} AdamWConfig;

void optim_adamw(GradTensor* gt, void* adamw_config);
AdamWConfig optim_adamw_get_config(f32 lr, f32 weight_decay);
void optim_adamw_step(AdamWConfig* config);

#endif
