#include "../include/tensor.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

Tensor* tensor_create(const u32* shape, usize shape_len, arena_allocator* arena) {
    if (shape_len > 4) {
        return NULL;
    }
    Tensor* t = arena_alloc(arena, sizeof(Tensor), 1);
    u32 curr_stride = 1;
    for (usize i = 0; i < shape_len; i++) {
        u32 dim = shape[shape_len - i - 1];
        t->shape[3-i] = dim;
        t->stride[3-i] = dim == 1 ? 0 : curr_stride;
        curr_stride *= dim;
    }

    for (usize i = shape_len; i < 4; i++) {
        t->shape[3-i] = 1;
        t->stride[3-i] = 0;
    }
    t->data_len = curr_stride;
    t->data = arena_alloc(arena, sizeof(f32), curr_stride);
    return t;
}

void tensor_print(const Tensor* t, bool print_data) {
    printf("Shape: [");
    for (int i = 0; i < 4; i++) {
        printf(" %u", t->shape[i]);
    }
    printf(" ]\nStrides: [");
    for (int i = 0; i < 4; i++) {
        printf(" %u", t->stride[i]);
    }

    printf(" ]\n");
    if (print_data) {
        printf("Data: [");
        usize data_len = t->data_len;
        for (usize i = 0; i < data_len; i++) {
            printf(" %f", t->data[i]);
            if ((i > 0 && (i+1) % t->shape[3] == 0) || t->shape[3] == 1) {
                printf(" ;");
            }
        }
        printf(" ]\n");
    }
}

void tensor_randomize(Tensor* t, f32 min, f32 max) {
    usize data_len = t->data_len;
    for (usize i = 0; i < data_len; i++) {
        t->data[i] = random_f32(min, max);
    }   
}

void tensor_set(Tensor* t, f32 v) {
    for (usize i = 0; i < t->data_len; i++) {
        t->data[i] = v;
    }
}

Tensor* tensor_add(const Tensor* a, const Tensor* b, arena_allocator* arena) {
    u32 target_shape[4];
    for (int i = 0; i < 4; i++) {
        if (a->shape[i] == b->shape[i]) {
            target_shape[i] = a->shape[i];
        } else if (a->shape[i] == 1) {
            target_shape[i] = b->shape[i];
        } else if (b->shape[i] == 1) {
            target_shape[i] = a->shape[i];
        } else {
            return NULL;
        }
    }

    Tensor* result = tensor_create(target_shape, 4, arena);
    _tensor_kernel_add(a, b, result);
    return result;
}

Tensor* tensor_mul(const Tensor* a, const Tensor* b, arena_allocator* arena) {
    u32 target_shape[4];
    for (int i = 0; i < 2; i++) {
        if (a->shape[i] == b->shape[i]) {
            target_shape[i] = a->shape[i];
        } else if (a->shape[i] == 1) {
            target_shape[i] = b->shape[i];
        } else if (b->shape[i] == 1) {
            target_shape[i] = a->shape[i];
        } else {
            return NULL;
        }
    }

    if (a->shape[3] != b->shape[2]) {
        return NULL;
    }

    target_shape[2] = a->shape[2];
    target_shape[3] = b->shape[3];
    Tensor* result = tensor_create(target_shape, 4, arena);
    _tensor_kernel_mul(a, b, result);
    return result;
}

Tensor* tensor_mul_tr(const Tensor* a, const Tensor* b, bool at, bool bt, arena_allocator* arena) {
    u32 target_shape[4];
    for (int i = 0; i < 2; i++) {
        if (a->shape[i] == b->shape[i]) {
            target_shape[i] = a->shape[i];
        } else if (a->shape[i] == 1) {
            target_shape[i] = b->shape[i];
        } else if (b->shape[i] == 1) {
            target_shape[i] = a->shape[i];
        } else {
            return NULL;
        }
    }

    Tensor* result = NULL;
    if (at && !bt) {
        if (a->shape[2] != b->shape[2]) {
            return NULL;
        }
        target_shape[2] = a->shape[3];
        target_shape[3] = b->shape[3];
        result = tensor_create(target_shape, 4, arena);
        _tensor_kernel_mul_at(a, b, result);
    } else if (!at && bt) {
        if (a->shape[3] != b->shape[3]) {
            return NULL;
        } 
        target_shape[2] = a->shape[2];
        target_shape[3] = b->shape[2];
        result = tensor_create(target_shape, 4, arena);
        _tensor_kernel_mul_bt(a, b, result);
    } else if (at && bt) {
        if (a->shape[2] != b->shape[3]) {
            return NULL;
        }
        target_shape[2] = a->shape[3];
        target_shape[3] = b->shape[2];
        result = tensor_create(target_shape, 4, arena);
        _tensor_kernel_mul_atbt(a, b, result);
    } else {
        if (a->shape[3] != b->shape[2]) {
            return NULL;
        }
        target_shape[2] = a->shape[2];
        target_shape[3] = b->shape[3];
        result = tensor_create(target_shape, 4, arena);
        _tensor_kernel_mul(a, b, result);
    }
    return result;
}

Tensor* tensor_reduce_add(const Tensor* src, usize dim, arena_allocator* arena) {
    if (dim > 3) {
        return NULL;
    }

    u32 res_shape[4];
    memcpy(res_shape, src->shape, 4 * sizeof(u32));
    res_shape[dim] = 1;
    Tensor* res = tensor_create(res_shape, 4, arena);
    _tensor_kernel_reduce_add(src, res, dim);
    return res;
}

Tensor* tensor_cross_entropy(const Tensor* src, const Tensor* truth, arena_allocator* arena) {
    if (src->shape[3] != truth->shape[3]) {
        return NULL;
    }

    if (truth->shape[0] != 1 || truth->shape[1] != 1) {
        return NULL;
    }

    if (src->shape[0] != 1 || src->shape[1] != 1) { // src->shape[2] can be != 1 for batches
        return NULL;
    }
    
    u32 shape[4] = {1, 1, 1, 1};
    Tensor* t = tensor_create(shape, 4, arena);
    _tensor_kernel_cross_entropy(src, truth, t);
    return t;
}

Tensor* tensor_sub_scaled(const Tensor* a, const Tensor* b, f32 alpha, arena_allocator* arena) {
    for (usize i = 0; i <  4; i++) {
        if (a->shape[i] != b->shape[i]) {
            return NULL;
        }
    }

    Tensor* result = tensor_create(a->shape, 4, arena);
    _tensor_kernel_sub_scaled(a, b, alpha, result);
    return result;
}
