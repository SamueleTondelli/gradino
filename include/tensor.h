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

void _add_tensor_kernel(const Tensor* a, const Tensor* b, Tensor* result);
Tensor* add_tensor(const Tensor* a, const Tensor* b);
void _add_tensor_bwd_kernel(Tensor* a_grad, Tensor* b_grad, const Tensor* in_grad);
void _mul_tensor_kernel(const Tensor* a, const Tensor* b, Tensor* result);
Tensor* mul_tensor(const Tensor* a, const Tensor* b);
void _mul_tensor_at_kernel(const Tensor* a, const Tensor* b, Tensor* result);
void _mul_tensor_bt_kernel(const Tensor* a, const Tensor* b, Tensor* result);
void _mul_tensor_atbt_kernel(const Tensor* a, const Tensor* b, Tensor* result);
Tensor* mul_tensor_tr(const Tensor* a, const Tensor* b, bool at, bool bt);

void _relu_tensor_kernel(const Tensor* src, Tensor* dst);
void _relu_bwd_tensor_kernel(const Tensor* src, Tensor* src_grad, const Tensor* in_grad);

#endif
