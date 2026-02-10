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

Tensor* tensor_create(u32* shape, usize shape_len);
void tensor_free(Tensor* t);

void tensor_print(const Tensor* t, bool print_data);
void tensor_randomize(Tensor* t, f32 min, f32 max);
void tensor_set(Tensor* t, f32 v);

Tensor* tensor_add(const Tensor* a, const Tensor* b);
Tensor* tensor_mul(const Tensor* a, const Tensor* b);
Tensor* tensor_mul_tr(const Tensor* a, const Tensor* b, bool at, bool bt);
Tensor* tensor_reduce_add(const Tensor* src, usize dim);

void _tensor_kernel_add(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_add_bwd(Tensor* a_grad, Tensor* b_grad, const Tensor* in_grad);
void _tensor_kernel_mul_at(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_mul_bt(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_mul_atbt(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_mul(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_reduce_add(const Tensor* src, Tensor* result, usize red_dim);
void _tensor_kernel_relu(const Tensor* src, Tensor* dst);
void _tensor_kernel_relu_bwd(const Tensor* src, Tensor* src_grad, const Tensor* in_grad);

#endif
