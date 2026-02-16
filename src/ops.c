#include "../include/ops.h"
#include "../include/grad.h"


void op_fwd(Op* op) {
    if (op->type == Mono) {
        const GradTensor* src = op->op.mono.src;
        GradTensor* dst = op->op.mono.dst;
        op->op.mono.fwd(src, dst);
    } else {
        const GradTensor* src1 = op->op.bin.src1;
        const GradTensor* src2 = op->op.bin.src2;
        GradTensor* dst = op->op.bin.dst;
        op->op.bin.fwd(src1, src2, dst);
    }
}

void op_bwd(Op* op) {
    if (op->type == Mono) {
        GradTensor* src = op->op.mono.src;
        const GradTensor* dst = op->op.mono.dst;
        if (src != NULL && src->grad != NULL && dst->grad != NULL) {
            if (!src->_grad_computed) {
                op->op.mono.bwd(src, dst);
                src->_grad_computed = true;
            } else {
                // goofy shit
                Tensor* old_grad = src->grad;
                src->grad = tensor_create(src->grad->shape, 4, _gradt_get_arena());
                op->op.mono.bwd(src, dst);
                _tensor_kernel_add(old_grad, src->grad, src->grad);
            }
        }
    } else {
        GradTensor* src1 = op->op.bin.src1;
        GradTensor* src2 = op->op.bin.src2;
        const GradTensor* dst = op->op.bin.dst;
        if (src1->grad != NULL && dst->grad != NULL) {
            if (!src1->_grad_computed) {
                op->op.bin.bwd_src1(src1, src2, dst);
                src1->_grad_computed = true;
            } else {
                Tensor* old_grad = src1->grad;
                src1->grad = tensor_create(src1->grad->shape, 4, _gradt_get_arena());
                op->op.bin.bwd_src1(src1, src2, dst);
                _tensor_kernel_add(old_grad, src1->grad, src1->grad);
            }
        }
        if (src2->grad != NULL && dst->grad != NULL) {
            if (!src2->_grad_computed) {
                op->op.bin.bwd_src2(src1, src2, dst);
                src2->_grad_computed = true;
            } else {
                Tensor* old_grad = src2->grad;
                src2->grad = tensor_create(src2->grad->shape, 4, _gradt_get_arena());
                op->op.bin.bwd_src2(src1, src2, dst);
                _tensor_kernel_add(old_grad, src2->grad, src2->grad);
            }
        }
    }
}

static void nop_fwd(const GradTensor* src, GradTensor* dst) {}
static void nop_bwd(GradTensor* src, const GradTensor* dst) {}

void op_set_nop(Op* op, struct GradTensor_struct* dst) {
    op->type = Mono;
    op->op.mono.src = NULL;
    op->op.mono.dst = dst;
    op->op.mono.fwd = nop_fwd;
    op->op.mono.bwd = nop_bwd;
}

static void relu_fwd(const GradTensor* src, GradTensor* dst) {
    _tensor_kernel_relu(src->tens, dst->tens);
}

static void relu_bwd(GradTensor* src, const GradTensor* dst) {
    _tensor_kernel_relu_bwd(src->tens, src->grad, dst->grad);
}

void op_set_relu(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* dst) {
   op->type = Mono;
   op->op.mono.src = src;
   op->op.mono.dst = dst;
   op->op.mono.fwd = relu_fwd;
   op->op.mono.bwd = relu_bwd; 
}

static void add_fwd(const GradTensor* src1, const GradTensor* src2, GradTensor* dst) {
    _tensor_kernel_add(src1->tens, src2->tens, dst->tens);
}

static void add_bwd_src1(GradTensor* src1, const GradTensor* src2, const GradTensor* dst) {
    _tensor_kernel_add_bwd(src1->grad, dst->grad, _gradt_get_arena());
}

static void add_bwd_src2(const GradTensor* src1, GradTensor* src2, const GradTensor* dst) {
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
}

static void mul_fwd(const GradTensor* src1, const GradTensor* src2, GradTensor* dst) {
    _tensor_kernel_mul(src1->tens, src2->tens, dst->tens);
}

static void mul_bwd_src1(GradTensor* src1, const GradTensor* src2, const GradTensor* dst) {
    _tensor_kernel_mul_bwd_a(src1->tens, src1->grad, src2->tens, dst->grad, _gradt_get_arena());
}

static void mul_bwd_src2(const GradTensor* src1, GradTensor* src2, const GradTensor* dst) {
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
}

static void cse_fwd(const GradTensor* src, const GradTensor* truth, GradTensor* dst) {
    _tensor_kernel_cross_entropy(src->tens, truth->tens, dst->tens);
}

static void cse_bwd_src(GradTensor* src, const GradTensor* truth, const GradTensor* dst) {
    _tensor_kernel_cross_entropy_bwd(src->tens, truth->tens, src->grad);
}

static void cse_bwd_truth(const GradTensor* src, GradTensor* truth, const GradTensor* dst) {
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
}

static void mse_fwd(const GradTensor* src, const GradTensor* truth, GradTensor* dst) {
    _tensor_kernel_mean_squared_error(src->tens, truth->tens, dst->tens);
}

static void mse_bwd_src(GradTensor* src, const GradTensor* truth, const GradTensor* dst) {
    _tensor_kernel_mean_squared_error_bwd(src->tens, truth->tens, src->grad);
}

static void mse_bwd_truth(const GradTensor* src, GradTensor* truth, const GradTensor* dst) {
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
}

static void sigmoid_fwd(const GradTensor* src, GradTensor* dst) {
    _tensor_kernel_sigmoid(src->tens, dst->tens);
}

static void sigmoid_bwd(GradTensor* src, const GradTensor* dst) {
    _tensor_kernel_sigmoid_bwd(src->grad, dst->tens, dst->grad);
}

void op_set_sigmoid(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* dst) {
    op->type = Mono;
    op->op.mono.src = src;
    op->op.mono.dst = dst;
    op->op.mono.fwd = sigmoid_fwd;
    op->op.mono.bwd = sigmoid_bwd;
}

static void mul_elemwise_fwd(const GradTensor* src1, const GradTensor* src2, GradTensor* dst) {
    _tensor_kernel_mul_elemwise(src1->tens, src2->tens, dst->tens);
}

static void mul_elemwise_bwd_src1(GradTensor* src1, const GradTensor* src2, const GradTensor* dst) {
    _tensor_kernel_mul_elemwise(src2->tens, dst->grad, src1->grad);
}

static void mul_elemwise_bwd_src2(const GradTensor* src1, GradTensor* src2, const GradTensor* dst) {
    _tensor_kernel_mul_elemwise(src1->tens, dst->grad, src2->grad);
}

void op_set_mul_elemwise(Op* op, struct GradTensor_struct* src1, struct GradTensor_struct* src2, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src1;
    op->op.bin.src2 = src2;
    op->op.bin.fwd = mul_elemwise_fwd;
    op->op.bin.bwd_src1 = mul_elemwise_bwd_src1;
    op->op.bin.bwd_src2 = mul_elemwise_bwd_src2;
}

static void tanh_fwd(const GradTensor* src, GradTensor* dst) {
    _tensor_kernel_tanh(src->tens, dst->tens);
}

static void tanh_bwd(GradTensor* src, const GradTensor* dst) {
    _tensor_kernel_tanh_bwd(src->grad, dst->tens, dst->grad);
}

void op_set_tanh(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* dst) {
    op->type = Mono;
    op->op.mono.src = src;
    op->op.mono.dst = dst;
    op->op.mono.fwd = tanh_fwd;
    op->op.mono.bwd = tanh_bwd;
}
