#ifndef OPTIM_H
#define OPTIM_H

#include "grad.h"

typedef struct {
    f32 lr;
} SGDConfig;

void optim_sgd(GradTensor* gt, void* sgd_config);
SGDConfig optim_sgd_get_config(f32 lr);

#endif
