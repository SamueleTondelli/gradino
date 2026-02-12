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
        op->op.mono.bwd(src, dst);
    } else {
        GradTensor* src1 = op->op.bin.src1;
        GradTensor* src2 = op->op.bin.src2;
        const GradTensor* dst = op->op.bin.dst;
        op->op.bin.bwd(src1, src2, dst);
    }
}

static void nop_fwd(const GradTensor* src, GradTensor* dst) {}
static void nop_bwd(GradTensor* src, const GradTensor* dst) {}

void op_set_nop(Op* op) {
    op->type = Mono;
    op->op.mono.src = NULL;
    op->op.mono.dst = NULL;
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

static void add_bwd(GradTensor* src1, GradTensor* src2, const GradTensor* dst) {
    _tensor_kernel_add_bwd(src1->grad, src2->grad, dst->grad, _gradt_get_arena());
}

void op_set_add(Op* op, struct GradTensor_struct* src1, struct GradTensor_struct* src2, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src1;
    op->op.bin.src2 = src2;
    op->op.bin.dst = dst;
    op->op.bin.fwd = add_fwd;
    op->op.bin.bwd = add_bwd;
}

static void mul_fwd(const GradTensor* src1, const GradTensor* src2, GradTensor* dst) {
    _tensor_kernel_mul(src1->tens, src2->tens, dst->tens);
}

static void mul_bwd(GradTensor* src1, GradTensor* src2, const GradTensor* dst) {
    _tensor_kernel_mul_bwd(src1->tens, src1->grad, src2->tens, src2->grad, dst->grad, _gradt_get_arena());
}

void op_set_mul(Op* op, struct GradTensor_struct* src1, struct GradTensor_struct* src2, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src1;
    op->op.bin.src2 = src2;
    op->op.bin.dst = dst;
    op->op.bin.fwd = mul_fwd;
    op->op.bin.bwd = mul_bwd;
}

static void cse_fwd(const GradTensor* src, const GradTensor* truth, GradTensor* dst) {
    _tensor_kernel_cross_entropy(src->tens, truth->tens, dst->tens);
}

static void cse_bwd(GradTensor* src, GradTensor* truth, const GradTensor* dst) {
    _tensor_kernel_cross_entropy_bwd(src->tens, truth->tens, src->grad);
}

void op_set_cse(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* truth, struct GradTensor_struct* dst) {
    op->type = Binary;
    op->op.bin.src1 = src;
    op->op.bin.src2 = truth;
    op->op.bin.dst = dst;
    op->op.bin.fwd = cse_fwd;
    op->op.bin.bwd = cse_bwd;
}
