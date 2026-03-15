#include "../include/ops.h"
#include "../include/grad.h"


void op_fwd(Op* op) {
    if (op->type == Mono) {
        const GradTensor* src = op->op.mono.src;
        GradTensor* dst = op->op.mono.dst;
        op->op.mono.fwd(src, dst, op->args);
    } else {
        const GradTensor* src1 = op->op.bin.src1;
        const GradTensor* src2 = op->op.bin.src2;
        GradTensor* dst = op->op.bin.dst;
        op->op.bin.fwd(src1, src2, dst, op->args);
    }
}

void op_bwd(Op* op) {
    if (op->type == Mono) {
        GradTensor* src = op->op.mono.src;
        const GradTensor* dst = op->op.mono.dst;
        if (src != NULL && src->grad != NULL && dst->grad != NULL) {
            if (!src->_grad_computed) {
                op->op.mono.bwd(src, dst, op->args);
                src->_grad_computed = true;
            } else {
                Tensor* accum = src->grad;
                Tensor* delta = tensor_create(accum->shape, 4, _gradt_get_arena());
                tensor_set(delta, 0.0f);
                src->grad = delta;
                op->op.mono.bwd(src, dst, op->args);
                _tensor_kernel_add(accum, delta, accum);
                src->grad = accum;
            }
        }
    } else {
        GradTensor* src1 = op->op.bin.src1;
        GradTensor* src2 = op->op.bin.src2;
        const GradTensor* dst = op->op.bin.dst;
        if (src1->grad != NULL && dst->grad != NULL) {
            if (!src1->_grad_computed) {
                op->op.bin.bwd_src1(src1, src2, dst, op->args);
                src1->_grad_computed = true;
            } else {
                Tensor* accum = src1->grad;
                Tensor* delta = tensor_create(accum->shape, 4, _gradt_get_arena());
                tensor_set(delta, 0.0f);
                src1->grad = delta;
                op->op.bin.bwd_src1(src1, src2, dst, op->args);
                _tensor_kernel_add(accum, delta, accum);
                src1->grad = accum;
            }
        }
        if (src2->grad != NULL && dst->grad != NULL) {
            if (!src2->_grad_computed) {
                op->op.bin.bwd_src2(src1, src2, dst, op->args);
                src2->_grad_computed = true;
            } else {
                Tensor* accum = src2->grad;
                Tensor* delta = tensor_create(accum->shape, 4, _gradt_get_arena());
                tensor_set(delta, 0.0f);
                src2->grad = delta;
                op->op.bin.bwd_src2(src1, src2, dst, op->args);
                _tensor_kernel_add(accum, delta, accum);
                src2->grad = accum;
            }
        }
    }
}

static void nop_fwd(const GradTensor* src, GradTensor* dst, void* args) {}
static void nop_bwd(GradTensor* src, const GradTensor* dst, void* args) {}

void op_set_nop(Op* op, struct GradTensor_struct* dst) {
    op->type = Mono;
    op->op.mono.src = NULL;
    op->op.mono.dst = dst;
    op->op.mono.fwd = nop_fwd;
    op->op.mono.bwd = nop_bwd;
    op->args = NULL;
}

static void relu_fwd(const GradTensor* src, GradTensor* dst, void* args) {
    _tensor_kernel_relu(src->tens, dst->tens);
}

static void relu_bwd(GradTensor* src, const GradTensor* dst, void* args) {
    _tensor_kernel_relu_bwd(src->tens, src->grad, dst->grad);
}

void op_set_relu(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* dst) {
   op->type = Mono;
   op->op.mono.src = src;
   op->op.mono.dst = dst;
   op->op.mono.fwd = relu_fwd;
   op->op.mono.bwd = relu_bwd; 
    op->args = NULL;
}

static void add_fwd(const GradTensor* src1, const GradTensor* src2, GradTensor* dst, void* args) {
    _tensor_kernel_add(src1->tens, src2->tens, dst->tens);
}

static void add_bwd_src1(GradTensor* src1, const GradTensor* src2, const GradTensor* dst, void* args) {
    _tensor_kernel_add_bwd(src1->grad, dst->grad, _gradt_get_arena());
}

static void add_bwd_src2(const GradTensor* src1, GradTensor* src2, const GradTensor* dst, void* args) {
    _tensor_kernel_add_bwd(src2->grad, dst->grad, _gradt_get_arena());
}

void op_set_add(Op* op, struct GradTensor_struct* src1, struct GradTensor_struct* src2, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src1;
    op->op.bin.src2 = src2;
    op->op.bin.dst = dst;
    op->op.bin.fwd = add_fwd;
    op->op.bin.bwd_src1 = add_bwd_src1;
    op->op.bin.bwd_src2 = add_bwd_src2;
    op->args = NULL;
}

static void mul_fwd(const GradTensor* src1, const GradTensor* src2, GradTensor* dst, void* args) {
    _tensor_kernel_mul(src1->tens, src2->tens, dst->tens);
}

static void mul_bwd_src1(GradTensor* src1, const GradTensor* src2, const GradTensor* dst, void* args) {
    _tensor_kernel_mul_bwd_a(src1->tens, src1->grad, src2->tens, dst->grad, _gradt_get_arena());
}

static void mul_bwd_src2(const GradTensor* src1, GradTensor* src2, const GradTensor* dst, void* args) {
    _tensor_kernel_mul_bwd_b(src1->tens, src2->tens, src2->grad, dst->grad, _gradt_get_arena());
}

void op_set_mul(Op* op, struct GradTensor_struct* src1, struct GradTensor_struct* src2, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src1;
    op->op.bin.src2 = src2;
    op->op.bin.dst = dst;
    op->op.bin.fwd = mul_fwd;
    op->op.bin.bwd_src1 = mul_bwd_src1;
    op->op.bin.bwd_src2 = mul_bwd_src2;
    op->args = NULL;
}

static void cse_fwd(const GradTensor* src, const GradTensor* truth, GradTensor* dst, void* args) {
    _tensor_kernel_cross_entropy(src->tens, truth->tens, dst->tens);
}

static void cse_bwd_src(GradTensor* src, const GradTensor* truth, const GradTensor* dst, void* args) {
    _tensor_kernel_cross_entropy_bwd(src->tens, truth->tens, src->grad);
}

static void cse_bwd_truth(const GradTensor* src, GradTensor* truth, const GradTensor* dst, void* args) {
    // no backward pass for truth
}

void op_set_cse(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* truth, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src;
    op->op.bin.src2 = truth;
    op->op.bin.dst = dst;
    op->op.bin.fwd = cse_fwd;
    op->op.bin.bwd_src1 = cse_bwd_src;
    op->op.bin.bwd_src2 = cse_bwd_truth;
    op->args = NULL;
}

static void mse_fwd(const GradTensor* src, const GradTensor* truth, GradTensor* dst, void* args) {
    _tensor_kernel_mean_squared_error(src->tens, truth->tens, dst->tens);
}

static void mse_bwd_src(GradTensor* src, const GradTensor* truth, const GradTensor* dst, void* args) {
    _tensor_kernel_mean_squared_error_bwd(src->tens, truth->tens, src->grad);
}

static void mse_bwd_truth(const GradTensor* src, GradTensor* truth, const GradTensor* dst, void* args) {
    // no backward pass for truth
}

void op_set_mse(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* truth, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src;
    op->op.bin.src2 = truth;
    op->op.bin.dst = dst;
    op->op.bin.fwd = mse_fwd;
    op->op.bin.bwd_src1 = mse_bwd_src;
    op->op.bin.bwd_src2 = mse_bwd_truth;
    op->args = NULL;
}

static void sigmoid_fwd(const GradTensor* src, GradTensor* dst, void* args) {
    _tensor_kernel_sigmoid(src->tens, dst->tens);
}

static void sigmoid_bwd(GradTensor* src, const GradTensor* dst, void* args) {
    _tensor_kernel_sigmoid_bwd(src->grad, dst->tens, dst->grad);
}

void op_set_sigmoid(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* dst) {
    op->type = Mono;
    op->op.mono.src = src;
    op->op.mono.dst = dst;
    op->op.mono.fwd = sigmoid_fwd;
    op->op.mono.bwd = sigmoid_bwd;
    op->args = NULL;
}

static void mul_elemwise_fwd(const GradTensor* src1, const GradTensor* src2, GradTensor* dst, void* args) {
    _tensor_kernel_mul_elemwise(src1->tens, src2->tens, dst->tens);
}

static void mul_elemwise_bwd_src1(GradTensor* src1, const GradTensor* src2, const GradTensor* dst, void* args) {
    _tensor_kernel_mul_elemwise(src2->tens, dst->grad, src1->grad);
}

static void mul_elemwise_bwd_src2(const GradTensor* src1, GradTensor* src2, const GradTensor* dst, void* args) {
    _tensor_kernel_mul_elemwise(src1->tens, dst->grad, src2->grad);
}

void op_set_mul_elemwise(Op* op, struct GradTensor_struct* src1, struct GradTensor_struct* src2, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src1;
    op->op.bin.src2 = src2;
    op->op.bin.dst = dst;
    op->op.bin.fwd = mul_elemwise_fwd;
    op->op.bin.bwd_src1 = mul_elemwise_bwd_src1;
    op->op.bin.bwd_src2 = mul_elemwise_bwd_src2;
    op->args = NULL;
}

static void tanh_fwd(const GradTensor* src, GradTensor* dst, void* args) {
    _tensor_kernel_tanh(src->tens, dst->tens);
}

static void tanh_bwd(GradTensor* src, const GradTensor* dst, void* args) {
    _tensor_kernel_tanh_bwd(src->grad, dst->tens, dst->grad);
}

void op_set_tanh(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* dst) {
    op->type = Mono;
    op->op.mono.src = src;
    op->op.mono.dst = dst;
    op->op.mono.fwd = tanh_fwd;
    op->op.mono.bwd = tanh_bwd;
    op->args = NULL;
}

static void concat_fwd(const GradTensor* src1, const GradTensor* src2, GradTensor* dst, void* args) {
    u32 concat_dim = *(u32*)args;
    _tensor_kernel_concat(src1->tens, src2->tens, concat_dim, dst->tens);
}

static void concat_bwd_src1(GradTensor* src1, const GradTensor* src2, const GradTensor* dst, void* args) {
    u32 concat_dim = *(u32*)args;
    _tensor_kernel_concat_bwd_a(src1->grad, src2->tens, dst->grad, concat_dim);
}

static void concat_bwd_src2(const GradTensor* src1, GradTensor* src2, const GradTensor* dst, void* args) {
    u32 concat_dim = *(u32*)args;
    _tensor_kernel_concat_bwd_b(src1->tens, src2->grad, dst->grad, concat_dim);
}

void op_set_concat(Op* op, struct GradTensor_struct* src1, struct GradTensor_struct* src2, struct GradTensor_struct* dst, u32 concat_dim) {
    op->type = Binary;
    op->op.bin.src1 = src1;
    op->op.bin.src2 = src2;
    op->op.bin.dst = dst;
    op->op.bin.fwd = concat_fwd;
    op->op.bin.bwd_src1 = concat_bwd_src1;
    op->op.bin.bwd_src2 = concat_bwd_src2;
    arena_allocator* arena = _gradt_get_arena();
    u32* args = arena_alloc(arena, sizeof(u32), 1);
    *args = concat_dim;
    op->args = (void*)args;
}
