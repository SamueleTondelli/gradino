# Gradino
Toy CPU autograd engine with AVX512 acceleration, with fully working backward pass and optimizer step.

## Features
The engine is mainly used through the `GradTensor` struct, which supports as operations for auto-differentiation:
- Add
- Matrix multiplication
- Element-wise multiplication
- ReLU
- Sigmoid
- Tanh
- Cross-entropy
- Mean squared error
- Tensor concatenation

There are additional wrappers in `include/nn.h`, for stuff like linear layers and LSTMs.

For tensor allocation, arenas are used for better performances and overall usability, as a lot of allocations are grouped together, predictable and repeated.

Optimizers can be found in `include/optim.h` and currently SGD, SGD with momentum, Adam and AdamW are supported.

Here's an example of what the training loop looks like.
```c
AdamWConfig adamw_conf = optim_adamw_get_config(LEARNING_RATE, WEIGHT_DECAY);
for (u32 j = 0; j < train_batches_size; j++) {
    GradTensor* pred = model_forward(&model, train_batches[j]);
    GradTensor* loss = nn_cross_enropy_loss(pred, train_labels[j]);
    gradt_backward(loss, optim_adamw, &adamw_conf);
    optim_adamw_step(&adamw_conf);
    arena_free(epoch_arena);
}
```

## Examples
Two examples can be found under the `examples` directory, in particular:
- `iris.c` trains a simple FFNN on the Iris dataset
- `activities_recognition.c` trains a time series classificator on the human activities recognition dataset, the classifier is made of LSTM -> Linear -> ReLU -> Linear, with the following model and forward function
```c
typedef struct {
    LSTM lstm;
    LinearLayer hidden;
    LinearLayer output;
} Model;


static GradTensor* model_forward(Model* m, GradTensor* x) {
    GradTensor* t = nn_lstm_forward(&m->lstm, x);
    t = nn_linear_forward(&m->hidden, t);
    t = nn_relu(t);
    t = nn_linear_forward(&m->output, t);
    return t;
}
```
To build and run the examples:
```bash
make all
./examples/bin/<example>
```