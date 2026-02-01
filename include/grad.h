#ifndef GRAD_H
#define GRAD_H

#include "tensor.h"
#include "ops.h"
#include "utils.h"

typedef struct GradTensor_struct {
    Tensor* tens;
    Tensor* grad;
    Op op;  // op which generates this tensor (dst = this)
} GradTensor;

GradTensor* create_gradt(u32* shape, usize shape_len);
GradTensor* create_gradt_from_tens(Tensor* tens);
void free_gradt(GradTensor* gt);

GradTensor* relu(GradTensor* gt);
GradTensor* add(GradTensor* gt1, GradTensor* gt2);
void backward(GradTensor* gt);

#endif
