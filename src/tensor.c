#include "../include/tensor.h"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

Tensor* create_tensor(u32* shape, usize shape_len) {
    if (shape_len > 4) {
        return NULL;
    }
    Tensor* t = malloc(sizeof(Tensor));
    u32 curr_stride = 1;
    for (usize i = 0; i < shape_len; i++) {
        u32 dim = shape[shape_len - i - 1];
        t->shape[3-i] = dim;
        t->stride[3-i] = dim == 1 ? 0 : curr_stride;
        curr_stride *= dim;
    }

    for (usize i = shape_len; i < 4; i++) {
        t->shape[3-i] = 1;
        t->stride[3-i] = curr_stride;
    }
    t->data_len = curr_stride;
    printf("Allocating %d\n", curr_stride);
    t->data = malloc(sizeof(f32) * curr_stride);
    return t;
}

void free_tensor(Tensor* t) {
    free(t->data);
    free(t);
}

void print_tensor(const Tensor* t, bool print_data) {
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

void randomize_tensor(Tensor* t, f32 min, f32 max) {
    usize data_len = t->data_len;
    for (usize i = 0; i < data_len; i++) {
        t->data[i] = random_f32(min, max);
    }   
}

void set_tensor(Tensor* t, f32 v) {
    for (usize i = 0; i < t->data_len; i++) {
        t->data[i] = v;
    }
}

Tensor* add_tensor(const Tensor* a, const Tensor* b) {
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

    Tensor* result = create_tensor(target_shape, 4);
    _add_tensor_kernel(a, b, result);
    return result;
}

void _add_tensor_bwd_kernel(Tensor* a_grad, Tensor* b_grad, const Tensor* in_grad) {
    memcpy(a_grad->data, in_grad->data, a_grad->data_len);
    memcpy(b_grad->data, in_grad->data, b_grad->data_len);
}

Tensor* mul_tensor(const Tensor* a, const Tensor* b) {
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
    Tensor* result = create_tensor(target_shape, 4);
    _mul_tensor_kernel(a, b, result);
    return result;
}

Tensor* mul_tensor_tr(const Tensor* a, const Tensor* b, bool at, bool bt) {
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
        result = create_tensor(target_shape, 4);
        _mul_tensor_at_kernel(a, b, result);
    } else if (!at && bt) {
        if (a->shape[3] != b->shape[3]) {
            return NULL;
        } 
        target_shape[2] = a->shape[2];
        target_shape[3] = b->shape[2];
        result = create_tensor(target_shape, 4);
        _mul_tensor_bt_kernel(a, b, result);
    } else if (at && bt) {
        if (a->shape[2] != b->shape[3]) {
            return NULL;
        }
        target_shape[2] = a->shape[3];
        target_shape[3] = b->shape[2];
        result = create_tensor(target_shape, 4);
        _mul_tensor_atbt_kernel(a, b, result);
    } else {
        if (a->shape[3] != b->shape[2]) {
            return NULL;
        }
        target_shape[2] = a->shape[2];
        target_shape[3] = b->shape[3];
        result = create_tensor(target_shape, 4);
        _mul_tensor_kernel(a, b, result);
    }
    return result;
}
