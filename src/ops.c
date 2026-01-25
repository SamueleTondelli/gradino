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
    relu_tensor(src->tens, dst->tens);
}

static void relu_bwd(GradTensor* src, const GradTensor* dst) {
    relu_bwd_tensot(src->tens, src->grad, dst->grad);
}

void op_set_relu(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* dst) {
   op->type = Mono;
   op->op.mono.src = src;
   op->op.mono.dst = dst;
   op->op.mono.fwd = relu_fwd;
   op->op.mono.bwd = relu_bwd; 
}
