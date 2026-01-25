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
    return result;
}

// static void matmul(const f32* a, const f32* b, f32* res, u32 a_rows, u32 a_cols, u32 b_cols) {
//     for (u32 i = 0; i < a_rows; i++) {
//         for (u32 j = 0; j < b_cols; j++) {
//             f32 el = 0.0;
//             for (u32 k = 0; k < a_cols; k++) {
//                 el += a[i * a_cols + k] * b[k * b_cols + j];
//             }
//             res[i * b_cols + j] = el;
//         }
//     }
// }

// static void matmul_vec(const f32* a, const f32* b, f32* res, u32 a_rows, u32 a_cols, u32 b_cols) {
//     for (u32 i = 0; i < a_rows; i++) {
//         for (u32 j = 0; j < b_cols; j++) {
//             u32 k = 0;
//             i32 b_idxs[16];
//             __m512 a_vec, b_vec, el_vec;
//             el_vec = _mm512_setzero_ps();
//             for (; k < a_cols - 15; k += 16) {
//                 a_vec = _mm512_loadu_ps(&a[i * a_cols + k]);
//                 for (i32 bk = 0; bk < 16; bk++) {
//                     b_idxs[bk] = (bk + k) * b_cols + j;
//                 }
//                 __m512i b_idxs_vec = _mm512_loadu_si512(b_idxs);
//                 b_vec = _mm512_i32gather_ps(b_idxs_vec, b, 4);
//                 el_vec = _mm512_fmadd_ps(a_vec, b_vec, el_vec);
//             }

//             f32 el = _mm512_reduce_add_ps(el_vec);
//             for (; k < a_cols; k++) {
//                 el += a[i * a_cols + k] * b[k * b_cols + j];
//             }
//             res[i * b_cols + j] = el;
//         }
//     }
// }

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
    return result;
}

void relu_tensor(const Tensor* src, Tensor* dst) {
    for (usize i = 0; i < 4; i++) {
        if (src->shape[i] != dst->shape[i]) {
            printf("Bad shape in relu_tensor\n");
            return;
        }
    }

    for (usize i = 0; i < src->data_len; i++) {
        dst->data[i] = (src->data[i] > 0.0) ? src->data[i] : 0.0;
    }
}

void relu_bwd_tensot(const Tensor* src, Tensor* src_grad, const Tensor* in_grad) {
    for (usize i = 0; i < 4; i++) {
        if (src->shape[i] != src_grad->shape[i] || src_grad->shape[i] != in_grad->shape[i] || src->shape[i] != in_grad->shape[i]) {
            printf("Bad shape in relu_bwd_tensor\n");
            return;
        }
    }

    for (usize i = 0; i < src->data_len; i++) {
        src_grad->data[i] = (src->data[i] > 0.0) ? in_grad->data[i] : 0.0;
    }
}
