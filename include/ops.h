#ifndef OPS_H
#define OPS_H

struct GradTensor_struct;

typedef enum {
    Mono,
    Binary
} OpType;

typedef void(*mono_op_fwd)(const struct GradTensor_struct* src, struct GradTensor_struct* dst);
typedef void(*mono_op_bwd)(struct GradTensor_struct* src, const struct GradTensor_struct* dst);
typedef void(*bin_op_fwd)(const struct GradTensor_struct* src1, const struct GradTensor_struct* src2, struct GradTensor_struct* dst);
typedef void(*bin_op_bwd)(struct GradTensor_struct* src1, struct GradTensor_struct* src2, const struct GradTensor_struct* dst);

typedef struct {
    struct GradTensor_struct* src;
    struct GradTensor_struct* dst;
    mono_op_fwd fwd;
    mono_op_bwd bwd;
} MonoOp;

typedef struct {
    struct GradTensor_struct* src1;
    struct GradTensor_struct* src2;
    struct GradTensor_struct* dst;
    bin_op_fwd fwd;
    bin_op_bwd bwd;
} BinOp;

typedef struct {
    OpType type;
    union {
        MonoOp mono;
        BinOp bin;
    } op;
} Op;

void op_fwd(Op* op);
void op_bwd(Op* op);

void op_set_nop(Op* op);
void op_set_relu(Op* op, struct GradTensor_struct* src, struct GradTensor_struct* dst);

#endif
