#ifndef TENSOR_H
#define TENSOR_H

#include "utils.h"
#include "arena.h"

#include <stdbool.h>

typedef struct {
    u32 shape[4];
    u32 stride[4];
    usize data_len;
    f32* data;
} Tensor;

Tensor* tensor_create(const u32* shape, usize shape_len, arena_allocator* arena);
Tensor* tensor_copy(const Tensor* src, arena_allocator* arena);
Tensor** tensor_create_split_views(const Tensor* src, u32 split_dim, arena_allocator* arena);

void tensor_print(const Tensor* t, bool print_data);
void tensor_randomize(Tensor* t, f32 min, f32 max);
void tensor_randomize_gaussian(Tensor* t, f32 mean, f32 variance);
void tensor_set(Tensor* t, f32 v);
void tensor_set_buffer(Tensor* t, f32* buffer, usize length);

Tensor* tensor_add(const Tensor* a, const Tensor* b, arena_allocator* arena);
Tensor* tensor_mul(const Tensor* a, const Tensor* b, arena_allocator* arena);
Tensor* tensor_mul_tr(const Tensor* a, const Tensor* b, bool at, bool bt, arena_allocator* arena);
Tensor* tensor_reduce_add(const Tensor* src, usize dim, arena_allocator* arena);
Tensor* tensor_cross_entropy(const Tensor* src, const Tensor* truth, arena_allocator* arena);
// result = a - alpha * b, no broadcasting
Tensor* tensor_sub_scaled(const Tensor* a, const Tensor* b, f32 alpha, arena_allocator* arena);
// result = a + alpha * b, no broadcasting
Tensor* tensor_add_scaled(const Tensor* a, const Tensor* b, f32 alpha, arena_allocator* arena);
Tensor* tensor_mean_squared_error(const Tensor* src, const Tensor* truth, arena_allocator* arena);
Tensor* tensor_sigmoid(const Tensor* src, arena_allocator* arena);
// no broadcasting
Tensor* tensor_mul_elemwise(const Tensor* a, const Tensor* b, arena_allocator* arena);
Tensor* tensor_tanh(const Tensor* src, arena_allocator* arena);
Tensor* tensor_mul_scalar(const Tensor* src, f32 v, arena_allocator* arena);
Tensor* tensor_concat(const Tensor* a, const Tensor* b, u32 concat_dim, arena_allocator* arena);

void _tensor_kernel_add(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_add_bwd(Tensor* grad, const Tensor* result_grad, arena_allocator* arena);
void _tensor_kernel_mul_at(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_mul_bt(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_mul_atbt(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_mul(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_mul_bwd_a(const Tensor* a, Tensor* a_grad, const Tensor* b, const Tensor* result_grad, arena_allocator* arena);
void _tensor_kernel_mul_bwd_b(const Tensor* a, const Tensor* b, Tensor* b_grad, const Tensor* result_grad, arena_allocator* arena);
void _tensor_kernel_reduce_add(const Tensor* src, Tensor* result, usize red_dim);
void _tensor_kernel_relu(const Tensor* src, Tensor* dst);
void _tensor_kernel_relu_bwd(const Tensor* src, Tensor* src_grad, const Tensor* result_grad);
void _tensor_kernel_cross_entropy_bwd(const Tensor* src, const Tensor* truth, Tensor* src_grad);
void _tensor_kernel_cross_entropy(const Tensor* src, const Tensor* truth, Tensor* result);
void _tensor_kernel_sub_scaled(const Tensor* a, const Tensor* b, f32 alpha, Tensor* result);
void _tensor_kernel_add_scaled(const Tensor* a, const Tensor* b, f32 alpha, Tensor* result);
void _tensor_kernel_mean_squared_error(const Tensor* src, const Tensor* truth, Tensor* result);
void _tensor_kernel_mean_squared_error_bwd(const Tensor* src, const Tensor* truth, Tensor* src_grad);
void _tensor_kernel_sigmoid(const Tensor* src, Tensor* result);
void _tensor_kernel_mul_elemwise(const Tensor* a, const Tensor* b, Tensor* result);
void _tensor_kernel_sigmoid_bwd(Tensor* src_grad, const Tensor* result, const Tensor* result_grad);
void _tensor_kernel_tanh(const Tensor* src, Tensor* result);
void _tensor_kernel_tanh_bwd(Tensor* src_grad, const Tensor* result, const Tensor* result_grad);
void _tensor_kernel_mul_scalar(const Tensor* src, f32 v, Tensor* result);
void _tensor_kernel_adam_update(Tensor* param, const Tensor* m1_scaled, const Tensor* m2_scaled, f32 epsilon, f32 lr);
void _tensor_kernel_concat(const Tensor* a, const Tensor* b, u32 concat_dim, Tensor* result);
void _tensor_kernel_concat_bwd_a(Tensor* a_grad, const Tensor* b, const Tensor* result_grad, u32 concat_dim);
void _tensor_kernel_concat_bwd_b(const Tensor* a, Tensor* b_grad, const Tensor* result_grad, u32 concat_dim);

#endif
