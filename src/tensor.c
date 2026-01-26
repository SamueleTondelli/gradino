#include "../include/tensor.h"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <immintrin.h>

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

void _add_tensor_kernel(const Tensor* a, const Tensor* b, Tensor* result) {
    u32 index[4] = {0, 0, 0, 0};
    if (a->shape[3] == b->shape[3] && a->shape[3] >= 16) {
        usize total_rows = result->shape[0] * result->shape[1] * result->shape[2];
        usize row_idx = 0;
        usize vecs = result->shape[3] / 16; 
        while (row_idx < total_rows) {
            for (usize k = 0; k < vecs; k++) {
                u32 a_offset = 0, b_offset = 0, res_offset = 0;
                for (int i = 0; i < 4; i++) {
                    a_offset += index[i] * a->stride[i];
                    b_offset += index[i] * b->stride[i];
                    res_offset += index[i] * result->stride[i];
                }

                __m512 a_vec = _mm512_loadu_ps(&a->data[a_offset]);
                __m512 b_vec = _mm512_loadu_ps(&b->data[b_offset]);
                __m512 res_vec = _mm512_add_ps(a_vec, b_vec);
                _mm512_storeu_ps(&result->data[res_offset], res_vec);

                index[3] += 16;
            }

            for (usize k = index[3]; k < result->shape[3]; k++) {
                u32 a_offset = 0, b_offset = 0, res_offset = 0;
                for (int i = 0; i < 4; i++) {
                    a_offset += index[i] * a->stride[i];
                    b_offset += index[i] * b->stride[i];
                    res_offset += index[i] * result->stride[i];
                }

                result->data[res_offset] = a->data[a_offset] + b->data[b_offset];
            }
            
            index[3] = 0;
            for (int i = 2; i >= 0; i--) {
                index[i]++;
                if (index[i] < result->shape[i]) {
                    break;
                } else {
                    index[i] = 0;
                }
            }
            row_idx++;
        }
    } else {
        usize total_elems = result->data_len;
        usize el_idx = 0;
        while (el_idx < total_elems) {
            u32 a_offset = 0, b_offset = 0, res_offset = 0;
            for (int i = 0; i < 4; i++) {
                a_offset += index[i] * a->stride[i];
                b_offset += index[i] * b->stride[i];
                res_offset += index[i] * result->stride[i];
            }

            result->data[res_offset] = a->data[a_offset] + b->data[b_offset];

            for (int i = 3; i >= 0; i--) {
                index[i]++;
                if (index[i] < result->shape[i]) {
                    break;
                } else {
                    index[i] = 0;
                }
            }
            el_idx++;
        }
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

static inline void mmul_6x16(const f32* a, const f32* b, f32* res, u32 a_cols, u32 b_cols, u32 n, u32 m) {
    __m512 res_tile[6];
    for (i32 i = 0; i < 6; i++) {
        res_tile[i] = _mm512_setzero_ps();
    }

    __m512 a_vec, b_vec;
    if (m == 16) {
        for (u32 k = 0; k < a_cols; k++) {
            b_vec = _mm512_loadu_ps(&b[k * b_cols]);
            
            for (u32 i = 0; i < n; i++) {
                a_vec = _mm512_set1_ps(a[i * a_cols + k]);
                res_tile[i] = _mm512_fmadd_ps(a_vec, b_vec, res_tile[i]);
            }
        }
    
        for (u32 i = 0; i < n; i++) {
            _mm512_storeu_ps(&res[i * b_cols], res_tile[i]);
        }
    } else {
        __mmask16 mask = 0xFFFF >> (16 - m);
        for (u32 k = 0; k < a_cols; k++) {
            b_vec = _mm512_maskz_loadu_ps(mask, &b[k * b_cols]);
            
            for (u32 i = 0; i < n; i++) {
                a_vec = _mm512_set1_ps(a[i * a_cols + k]);
                res_tile[i] = _mm512_fmadd_ps(a_vec, b_vec, res_tile[i]);
            }
        }

        for (u32 i = 0; i < n; i++) {
            _mm512_mask_storeu_ps(&res[i * b_cols], mask, res_tile[i]);
        }
    }
}

// https://salykova.github.io/gemm-cpu
static void matmul_tile6x16(const f32* a, const f32* b, f32* res, u32 a_rows, u32 a_cols, u32 b_cols) {
    u32 n, m;
    for (u32 i = 0; i < a_rows; i += 6) {
        n = (a_rows - i) >= 6 ? 6 : (a_rows - i);
        for (u32 j = 0; j < b_cols; j += 16) {
            m = (b_cols - j) >= 16 ? 16 : (b_cols - j);
            // printf("mmul_6x16 at (%u, %u) (%u, %u): a[%u], b[%u], res[%u]\n", i, j, m, n, i*a_cols, j, i*b_cols + j);
            mmul_6x16(&a[i * a_cols], &b[j], &res[i * b_cols + j], a_cols, b_cols, n, m);
        }
    }
}

void _mul_tensor_kernel(const Tensor* a, const Tensor* b, Tensor* result) {
    u32 index[4] = {0, 0, 0, 0};
    usize mat_idx = 0, total_mats = result->shape[0] * result->shape[1];
    while (mat_idx < total_mats) {
        u32 a_offset = 0, b_offset = 0, res_offset = 0;
        for (int i = 0; i < 2; i++) {
            a_offset += index[i] * a->stride[i];
            b_offset += index[i] * b->stride[i];
            res_offset += index[i] * result->stride[i];
        }

        matmul_tile6x16(&a->data[a_offset], &b->data[b_offset], &result->data[res_offset], a->shape[2], a->shape[3], b->shape[3]);
    
        for (int i = 1; i >= 0; i--) {
            index[i]++;
            if (index[i] < result->shape[i]) {
                break;
            } else {
                index[i] = 0;
            }
        }
        mat_idx++;
    }
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

void _relu_tensor_kernel(const Tensor* src, Tensor* dst) {
    for (usize i = 0; i < 4; i++) {
        if (src->shape[i] != dst->shape[i]) {
            printf("Bad shape in relu_tensor\n");
            return;
        }
    }

    // for (usize i = 0; i < src->data_len; i++) {
    //     dst->data[i] = (src->data[i] > 0.0) ? src->data[i] : 0.0;
    // }

    u32 vecs = src->data_len / 16;
    usize i = 0;
    __m512 zerov = _mm512_setzero_ps();
    for (u32 iv = 0; iv < vecs; iv++) {
        __m512 srcv = _mm512_loadu_ps(&src->data[i]);
        srcv = _mm512_max_ps(srcv, zerov);
        _mm512_storeu_ps(&dst->data[i], srcv);
        i += 16;
    }

    for (; i < src->data_len; i++) {
        dst->data[i] = (src->data[i] > 0.0) ? src->data[i] : 0.0;
    }
}

void _relu_bwd_tensor_kernel(const Tensor* src, Tensor* src_grad, const Tensor* in_grad) {
    for (usize i = 0; i < 4; i++) {
        if (src->shape[i] != src_grad->shape[i] || src_grad->shape[i] != in_grad->shape[i] || src->shape[i] != in_grad->shape[i]) {
            printf("Bad shape in relu_bwd_tensor\n");
            return;
        }
    }

    // for (usize i = 0; i < src->data_len; i++) {
    //     src_grad->data[i] = (src->data[i] > 0.0) ? in_grad->data[i] : 0.0;
    // }
    u32 vecs = src->data_len / 16;
    usize i = 0;
    __m512 zerov = _mm512_setzero_ps();
    for (u32 iv = 0; iv < vecs; iv++) {
        __m512 srcv = _mm512_loadu_ps(&src->data[i]);
        __m512 ingv = _mm512_loadu_ps(&in_grad->data[i]);
        __mmask16 mask = _mm512_cmp_ps_mask(srcv, zerov, _CMP_GT_OQ);
        ingv = _mm512_maskz_mov_ps(mask, ingv);
        _mm512_storeu_ps(&src_grad->data[i], ingv);
        i += 16;
    }
    
    for (; i < src->data_len; i++) {
        src_grad->data[i] = (src->data[i] > 0.0) ? in_grad->data[i] : 0.0;
    }
}

void _mul_tensor_at_kernel(const Tensor* a, const Tensor* b, Tensor* result) {
    UNIMPL();
}

static void matmul_bt(const f32* a, const f32* b, f32* res, u32 a_rows, u32 a_cols, u32 b_cols) {
    u32 vecs = a_cols / 16;
    for (u32 i = 0; i < a_rows; i++) {
        for (u32 j = 0; j < b_cols; j++) {
            u32 k = 0;
            __m512 av, bv;
            __m512 acc = _mm512_setzero_ps();
            for (u32 kv = 0; kv < vecs; kv++) {
                // printf("i: %d, j: %d, k: %d, a[%d], b[%d]\n", i, j, k, i*a_cols + k, j*a_cols + k);
                av = _mm512_loadu_ps(&a[i * a_cols + k]);
                bv = _mm512_loadu_ps(&b[j * a_cols + k]);
                acc = _mm512_fmadd_ps(av, bv, acc);
                k += 16;
            }
            // printf("finished vec sumadd\n");
            f32 el = _mm512_reduce_add_ps(acc);
            for (; k < a_cols; k++) {
                // printf("i: %d, j: %d, k: %d, a[%d], b[%d]\n", i, j, k, i*a_cols + k, j*a_cols + k);
                el += a[i * a_cols + k] * b[j * a_cols + k];
            }
            // printf("res[%d] = %f\n", i*b_cols + j, el);
            res[i * b_cols + j] = el;
        }
    }
}

void _mul_tensor_bt_kernel(const Tensor* a, const Tensor* b, Tensor* result) {
    u32 index[4] = {0, 0, 0, 0};
    usize mat_idx = 0, total_mats = result->shape[0] * result->shape[1];
    while (mat_idx < total_mats) {
        u32 a_offset = 0, b_offset = 0, res_offset = 0;
        for (int i = 0; i < 2; i++) {
            a_offset += index[i] * a->stride[i];
            b_offset += index[i] * b->stride[i];
            res_offset += index[i] * result->stride[i];
        }

        matmul_bt(&a->data[a_offset], &b->data[b_offset], &result->data[res_offset], a->shape[2], a->shape[3], b->shape[2]);
            
        for (int i = 1; i >= 0; i--) {
            index[i]++;
            if (index[i] < result->shape[i]) {
                break;
            } else {
                index[i] = 0;
            }
        }
        mat_idx++;
    }
}

void _mul_tensor_atbt_kernel(const Tensor* a, const Tensor* b, Tensor* result) {
    UNIMPL();
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
