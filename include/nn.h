#ifndef NN_H
#define NN_H

#include "grad.h"

typedef struct {
    GradTensor* w;
    GradTensor* b;
    GradTensor* _proj;
} LinearLayer;

LinearLayer nn_linear_create(u32 in, u32 out);
GradTensor* nn_linear_forward(LinearLayer* layer, GradTensor* in);
GradTensor* nn_relu(GradTensor* gt);
GradTensor* nn_cross_enropy_loss(GradTensor* src, GradTensor* truth);
GradTensor* nn_mean_squared_error_loss(GradTensor* src, GradTensor* truth);
GradTensor* nn_sigmoid(GradTensor* src);
GradTensor* nn_tanh(GradTensor* src);

typedef struct {
    LinearLayer in_gate;
    LinearLayer in_val;
    LinearLayer for_gate;
    LinearLayer out_gate;
    u32 in_size;
    u32 hidden_size;
    GradTensor** hidden_states;
    GradTensor** cec_states;
} LSTM;

LSTM nn_lstm_init(u32 in, u32 hidden);
GradTensor* nn_lstm_forward(LSTM* lstm, GradTensor* in);

#endif
