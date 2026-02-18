#include "../include/nn.h"

LinearLayer nn_linear_create(u32 in, u32 out) {
    u32 w_shape[4] = {1, 1, in, out};
    u32 b_shape[4] = {1, 1, 1, out};
    LinearLayer l = {
        .w = gradt_create(w_shape, 4),
        .b = gradt_create(b_shape, 4),
        ._proj = NULL
    };
    gradt_enable_optim(l.w);
    gradt_enable_optim(l.b);

    f32 variance = 2.0 / ((f32)in + out);
    tensor_randomize_gaussian(l.w->tens, 0.0, variance);
    tensor_set(l.b->tens, 0.0);
    return l;
}

GradTensor* nn_linear_forward(LinearLayer* layer, GradTensor* in) {
    layer->_proj = gradt_mul(in, layer->w);
    return gradt_add(layer->_proj, layer->b);
}

GradTensor* nn_relu(GradTensor* gt) {
    return gradt_relu(gt);
}

GradTensor* nn_cross_enropy_loss(GradTensor* src, GradTensor* truth) {
    return gradt_cross_entropy_loss(src, truth);
}

GradTensor* nn_mean_squared_error_loss(GradTensor* src, GradTensor* truth) {
    return gradt_mean_squared_error_loss(src, truth);
}

GradTensor* nn_sigmoid(GradTensor* src) {
    return gradt_sigmoid(src);
}

GradTensor* nn_tanh(GradTensor* src) {
    return gradt_tanh(src);
}

LSTM nn_lstm_init(u32 in, u32 hidden) {
    LSTM l = {
        .in_gate = nn_linear_create(in + hidden, hidden),
        .in_val = nn_linear_create(in + hidden, hidden),
        .for_gate = nn_linear_create(in + hidden, hidden),
        .out_gate = nn_linear_create(in + hidden, hidden),
        .in_size = in,
        .hidden_size = hidden
    };
    return l;
}

GradTensor* nn_lstm_forward(LSTM* lstm, GradTensor* in) {
    // in shape (1, Step, Batch, Feat)
    u32 n_step = in->tens->shape[1];
    
    // steps Step * (Batch, Feat)
    GradTensor** steps = gradt_create_split_views(in, 1);

    u32 hidden_shape[4] = {1, 1, in->tens->shape[2], lstm->hidden_size};
    GradTensor* hs = gradt_create(hidden_shape, 4);
    tensor_set(hs->tens, 0.0);
    GradTensor* cec = gradt_create(hidden_shape, 4);
    tensor_set(cec->tens, 0.0);
    for (u32 i = 0; i < n_step; i++) {
        GradTensor* in_hs = gradt_concat(steps[i], hs, 3);

        GradTensor* f_t = nn_linear_forward(&lstm->for_gate, in_hs);
        f_t = nn_sigmoid(f_t);
        cec = gradt_mul_elemwise(cec, f_t);

        GradTensor* i_t = nn_linear_forward(&lstm->in_gate, in_hs);
        i_t = nn_sigmoid(i_t);
        GradTensor* i_val_t = nn_linear_forward(&lstm->in_val, in_hs);
        i_val_t = nn_tanh(i_val_t);
        GradTensor* cec_update = gradt_mul_elemwise(i_val_t, i_t);
        cec = gradt_add(cec, cec_update);

        GradTensor* o_t = nn_linear_forward(&lstm->out_gate, in_hs);
        o_t = nn_sigmoid(o_t);
        GradTensor* new_hs = nn_tanh(cec);

        hs = gradt_mul_elemwise(new_hs, o_t);
    }
    
    return hs;
}
