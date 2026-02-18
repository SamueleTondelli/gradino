#ifndef TEST_H
#define TEST_H

#include "utils.h"

void test_add(u32 rows, u32 cols);
void test_mul(u32 m, u32 k, u32 n);
void test_reduce_add(u32 rows, u32 cols, u32 dim);
void test_arena(usize reserve, usize commit, usize alloc_size, u32 n_allocs);
void test_grad_relu();
void test_grad_bwd();
void test_xor(f32 lr, f32 weight_decay, u32 hidden_size, u32 epochs);
void test_bwd_perf(f32 lr, u32 hidden_size, u32 bs, u32 io_dim, u32 n_batches, u32 epochs);
void test_split_views();
void test_concat();

#endif
