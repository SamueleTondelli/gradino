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

GradTensor* gradt_create(u32* shape, usize shape_len);
GradTensor* gradt_create_from_tens(Tensor* tens);
void gradt_free(GradTensor* gt);

GradTensor* gradt_relu(GradTensor* gt);
GradTensor* gradt_add(GradTensor* gt1, GradTensor* gt2);
GradTensor* gradt_mul(GradTensor* gt1, GradTensor* gt2);
void gradt_backward(GradTensor* gt);

#endif
