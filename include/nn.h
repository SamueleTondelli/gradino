#ifndef NN_H
#define NN_H

#include "grad.h"

typedef struct {
    GradTensor* w;
    GradTensor* b;
    GradTensor* _proj;
} LinearLayer;

LinearLayer nn_linear_create(u32 in, u32 out);
GradTensor* nn_linear_forward(LinearLayer* layer, GradTensor* in);
GradTensor* nn_relu(GradTensor* gt);
GradTensor* nn_cross_enropy_loss(GradTensor* src, GradTensor* truth);

#endif
