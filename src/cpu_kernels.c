#include "../include/tensor.h"

#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <x86intrin.h>
#include <string.h>

void _tensor_kernel_add(const Tensor* a, const Tensor* b, Tensor* result) {
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

void _tensor_kernel_add_bwd(Tensor* a_grad, Tensor* b_grad, const Tensor* in_grad, arena_allocator* arena) {
    Tensor* a_red = in_grad;
    Tensor* b_red = in_grad;

    for (usize i = 0; i < 4; i++) {
        if (a_grad->shape[i] < b_grad->shape[i]) {
            a_red = tensor_reduce_add(a_red, i, arena);
        } else if (b_grad->shape[i] < a_grad->shape[i]) {
            b_red = tensor_reduce_add(b_red, i, arena);
        }
     }
    
    memcpy(a_grad->data, a_red->data, a_grad->data_len * sizeof(f32));
    memcpy(b_grad->data, b_red->data, b_grad->data_len * sizeof(f32));
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

void _tensor_kernel_mul(const Tensor* a, const Tensor* b, Tensor* result) {
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
void _tensor_kernel_relu(const Tensor* src, Tensor* dst) {
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

void _tensor_kernel_relu_bwd(const Tensor* src, Tensor* src_grad, const Tensor* in_grad) {
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

static void mmul_6x16_at(const f32* a, const f32* b, f32* res, u32 at_rows, u32 b_rows, u32 b_cols, int n, int m) {
    __m512 res_tile[6];
    for (u32 i = 0; i < 6; i++) {
        res_tile[i] = _mm512_setzero_ps();
    }

    if (m == 16) {
        for (u32 k = 0; k < b_rows; k++) {
            __m512 b_vec = _mm512_loadu_ps(&b[k * b_cols]);

            for (u32 i = 0; i < n; i++) {
                __m512 a_vec = _mm512_set1_ps(a[at_rows * k + i]);
                res_tile[i] = _mm512_fmadd_ps(a_vec, b_vec, res_tile[i]);
            }
        }

        for (u32 i = 0; i < n; i++) {
            _mm512_storeu_ps(&res[i * b_cols], res_tile[i]);
        }
    } else {
        __mmask16 mask = 0xFFFF >> (16 - m);
        for (u32 k = 0; k < b_rows; k++) {
            __m512 b_vec = _mm512_maskz_loadu_ps(mask, &b[k * b_cols]);

            for (u32 i = 0; i < n; i++) {
                __m512 a_vec = _mm512_set1_ps(a[at_rows * k + i]);
                res_tile[i] = _mm512_fmadd_ps(a_vec, b_vec, res_tile[i]);
            }
        }

        for (u32 i = 0; i < n; i++) {
            _mm512_storeu_ps(&res[i * b_cols], res_tile[i]);
        }
    }
}

static void matmul_at(const f32* a, const f32* b, f32* res, u32 at_rows, u32 at_cols, u32 b_cols) {
    for (u32 i = 0; i < at_rows; i += 6) {
        u32 n = (at_rows - i) >= 6 ? 6 : (at_rows - i);
        for (u32 j = 0; j < b_cols; j += 16) {
            u32 m = (b_cols - j) >= 16 ? 16 : (b_cols - j);
            mmul_6x16_at(&a[i], &b[j], &res[i * b_cols + j], at_rows, at_cols, b_cols, n, m);
        }
    }
}

void _tensor_kernel_mul_at(const Tensor* a, const Tensor* b, Tensor* result) {
    u32 index[4] = {0, 0, 0, 0};
    usize mat_idx = 0, total_mats = result->shape[0] * result->shape[1];
    while (mat_idx < total_mats) {
        u32 a_offset = 0, b_offset = 0, res_offset = 0;
        for (int i = 0; i < 2; i++) {
            a_offset += index[i] * a->stride[i];
            b_offset += index[i] * b->stride[i];
            res_offset += index[i] * result->stride[i];
        }

        matmul_at(&a->data[a_offset], &b->data[b_offset], &result->data[res_offset], a->shape[3], a->shape[2], b->shape[3]);
            
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

void _tensor_kernel_mul_bt(const Tensor* a, const Tensor* b, Tensor* result) {
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

void _tensor_kernel_mul_atbt(const Tensor* a, const Tensor* b, Tensor* result) {
    UNIMPL();
}

void _tensor_kernel_mul_bwd(const Tensor* a, Tensor* a_grad, const Tensor* b, Tensor* b_grad, const Tensor* result_grad, arena_allocator* arena) {
    if (result_grad->shape[0] == a_grad->shape[0] && result_grad->shape[1] == a_grad->shape[1]) {
        _tensor_kernel_mul_bt(result_grad, b, a_grad);
    } else {
        Tensor* a_grad_broad = tensor_mul_tr(result_grad, b, false, true, arena);
        for (usize i = 0; i < 2; i++) {
            if (result_grad->shape[i] != a_grad->shape[i]) {
                a_grad_broad = tensor_reduce_add(a_grad_broad, i, arena);
            }
        }
        memcpy(a_grad->data, a_grad_broad->data, a_grad->data_len * sizeof(f32));
    }

    if (result_grad->shape[0] == b_grad->shape[0] && result_grad->shape[1] == b_grad->shape[1]) {
        _tensor_kernel_mul_at(a, result_grad, b_grad);
    } else {
        Tensor* b_grad_broad = tensor_mul_tr(a, result_grad, true, false, arena);
        for (usize i = 0; i < 2; i++) {
            if (result_grad->shape[i] != b_grad->shape[i]) {
                b_grad_broad = tensor_reduce_add(b_grad, i, arena);
            }
        }
        memcpy(b_grad->data, b_grad_broad->data, b_grad->data_len * sizeof(f32));
    }
}

void _tensor_kernel_reduce_add(const Tensor* src, Tensor* result, usize red_dim) {
    usize dims_common[3];
    usize dims_map[3];
    usize dims_idx = 0;
    for (usize i = 0; i < 4; i++) {
        if (i != red_dim) {
            dims_common[dims_idx] = src->shape[i];
            dims_map[dims_idx] = i;
            dims_idx++;
        }
    }

    for (usize i = 0; i < dims_common[0]; i++) {
        for (usize j = 0; j < dims_common[1]; j++) {
            for (usize k = 0; k < dims_common[2]; k++) {
                usize src_offset = 0;
                src_offset += i * src->stride[dims_map[0]];
                src_offset += j * src->stride[dims_map[1]];
                src_offset += k * src->stride[dims_map[2]];
                usize res_offset = 0;
                res_offset += i * result->stride[dims_map[0]];
                res_offset += j * result->stride[dims_map[1]];
                res_offset += k * result->stride[dims_map[2]];
                result->data[res_offset] = 0.0;
                for (usize r = 0; r < src->shape[red_dim]; r++) {
                    result->data[res_offset] += src->data[src_offset];
                    src_offset += src->stride[red_dim];
                }
            }
        }
    }
}

void _tensor_kernel_cross_entropy(const Tensor* src, const Tensor* truth, Tensor* result) {
    f32 ce = 0.0;
    for (usize j = 0; j < src->shape[2]; j++) {
        f32 acc = 0.0;
        usize base = src->stride[2] * j;
        for (usize i = 0; i < src->data_len; i++) {
            usize idx = base + i;
            if (truth->data[idx] >= 0.99) { // fp stuff
                ce = -src->data[idx];
            }
            acc += expf(src->data[idx]);
        }
        ce += logf(acc) / (f32)src->shape[2];
    }
    result->data[0] = ce;
}

void _tensor_kernel_cross_entropy_bwd(const Tensor* src, const Tensor* truth, Tensor* src_grad) {
    for (usize j = 0; j < src->shape[2]; j++) {
        f32 sftmx_den = 0.0;
        usize base = src->stride[2] * j;
        for (usize i = 0; i < src->shape[3]; i++) {
            usize idx = base + i;
            sftmx_den += expf(src->data[idx]);
        }

        for (usize i = 0; i < src->shape[3]; i++) {
            usize idx = base + i;
            src_grad->data[idx] = (expf(src->data[idx])/sftmx_den - truth->data[idx]) / (f32)src->shape[2];
        }
    }
}

