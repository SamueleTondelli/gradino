#ifndef TENSOR_H
#define TENSOR_H

#include "utils.h"

#include <stdbool.h>

typedef struct {
    u32 shape[4];
    u32 stride[4];
    usize data_len;
    f32* data;
} Tensor;

Tensor* create_tensor(u32* shape, usize shape_len);
void free_tensor(Tensor* t);

void print_tensor(const Tensor* t, bool print_data);
void randomize_tensor(Tensor* t, f32 min, f32 max);

Tensor* add_tensor(const Tensor* a, const Tensor* b);
Tensor* mul_tensor(const Tensor* a, const Tensor* b);

void relu_tensor(const Tensor* src, Tensor* dst);
void relu_bwd_tensot(const Tensor* src, Tensor* src_grad, const Tensor* in_grad);

#endif
