#ifndef GRAD_H
#define GRAD_H

#include "tensor.h"
#include "ops.h"
#include "utils.h"
#include "arena.h"

typedef struct GradTensor_struct {
    Tensor* tens;
    Tensor* grad;
    Tensor* prev_grad;
    Op op; // op which generates this tensor (dst = this)
    bool optimize;
    bool _grad_computed; // grad already computed in bwd pass, so add instead of replacing
    // Adam stuff
    Tensor* _first_moment;
    Tensor* _second_moment;
} GradTensor;

typedef void(*Optimizer)(GradTensor* gt, void* optim_config);

void gradt_set_arena(arena_allocator* arena);
void gradt_destroy_arena();
void gradt_detach_arena();
void gradt_set_and_destroy_arena(arena_allocator* arena);
void gradt_free_arena();
arena_allocator* _gradt_get_arena();

GradTensor* gradt_create(u32* shape, usize shape_len);
GradTensor* gradt_create_from_tens(Tensor* tens);
GradTensor* gradt_create_from_labels(u32* labels, u32 n_classes, u32 n_labels);
GradTensor* gradt_create_nograd(u32* shape, usize shape_len);
GradTensor** gradt_create_split_views(const GradTensor* gt, u32 split_dim);
void gradt_enable_optim(GradTensor* gt);

GradTensor* gradt_relu(GradTensor* gt);
GradTensor* gradt_add(GradTensor* gt1, GradTensor* gt2);
GradTensor* gradt_mul(GradTensor* gt1, GradTensor* gt2);
GradTensor* gradt_cross_entropy_loss(GradTensor* src, GradTensor* truth);
GradTensor* gradt_mean_squared_error_loss(GradTensor* src, GradTensor* truth);
GradTensor* gradt_sigmoid(GradTensor* src);
GradTensor* gradt_mul_elemwise(GradTensor* gt1, GradTensor* gt2);
GradTensor* gradt_tanh(GradTensor* src);
GradTensor* gradt_concat(GradTensor* gt1, GradTensor* gt2, u32 concat_dim);

void gradt_backward(GradTensor* gt, Optimizer optim, void* optim_config);
f32 gradt_compute_accuracy(GradTensor* src, GradTensor* truth);

#endif
