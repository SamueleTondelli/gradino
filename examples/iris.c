#include "../include/grad.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include <string.h>

#define IRIS_CSV_PATH "examples/datasets/IRIS.csv"
#define IRIS_MAX_LINE 256

const f32 TEST_SPLIT = 0.2;
const f32 VAL_SPLIT = 0.2;
const u32 BATCH_SIZE = 16;
const u32 HIDDEN_SIZE = 64;
const u32 EPOCHS = 10;
const f32 LEARNING_RATE = 1e-2;
const f32 WEIGHT_DECAY = 1e-5;

typedef enum: u32 {
    SETOSA = 0,
    VERSICOLOR = 1,
    VIRGINICA = 2
} IrisClass;

typedef struct {
    f32 sepal_length;
    f32 sepal_width;
    f32 petal_length;
    f32 petal_width;
    IrisClass class;
} IrisEntry;

static IrisClass parse_species(const char* species);
static u32 count_lines(FILE* f);
static IrisEntry* load_iris(arena_allocator* arena, const char* path, u32* out_count);
static IrisEntry* shuffle_iris(arena_allocator* arena, IrisEntry* entries, u32 count);
static GradTensor** create_batches(IrisEntry* entries, u32 count, u32* out_n_batches, arena_allocator* arena);
static GradTensor** create_label_batches(IrisEntry* entries, u32 count, u32* out_n_batches, arena_allocator* arena);

typedef struct {
    LinearLayer h1;
    LinearLayer h2;
    LinearLayer output;
} Model;

static Model model_create(u32 hidden_size);
static GradTensor* model_forward(Model* m, GradTensor* x);

int main() {
    printf("Iris dataset example\n");
    arena_allocator* permanent_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(permanent_arena);

    u32 count;
    IrisEntry* data = load_iris(permanent_arena, IRIS_CSV_PATH, &count);
    printf("Loaded %u entries\n", count);

    IrisEntry* shuffled = shuffle_iris(permanent_arena, data, count);

    usize test_size = (f32)count * TEST_SPLIT;
    printf("Test set size: %lu\n", test_size);
    IrisEntry* test = shuffled;
    usize val_size = (f32)count * VAL_SPLIT;
    printf("Validation set size: %lu\n", val_size);
    IrisEntry* val = &shuffled[test_size];
    usize train_size = count - test_size - val_size;
    printf("Train set size: %lu\n", train_size);
    IrisEntry* train = &shuffled[test_size + val_size];

    u32 test_batches_size;
    GradTensor** test_batches = create_batches(test, test_size, &test_batches_size, permanent_arena);
    u32 test_label_batches_size;
    GradTensor** test_labels = create_label_batches(test, test_size, &test_label_batches_size, permanent_arena);
    printf("Test batches: %u, Label batches: %u\n", test_batches_size, test_label_batches_size);
    u32 val_batches_size;
    GradTensor** val_batches = create_batches(val, val_size, &val_batches_size, permanent_arena);
    u32 val_label_batches_size;
    GradTensor** val_labels = create_label_batches(val, val_size, &val_label_batches_size, permanent_arena);
    printf("Validation batches: %u, Label batches: %u\n", val_batches_size, val_label_batches_size);
    u32 train_batches_size;
    GradTensor** train_batches = create_batches(train, train_size, &train_batches_size, permanent_arena);
    u32 train_label_batches_size;
    GradTensor** train_labels = create_label_batches(train, train_size, &train_label_batches_size, permanent_arena);
    printf("Train batches: %u, Label batches: %u\n", train_batches_size, train_label_batches_size);

    printf("Creating model with hidden size %u\n", HIDDEN_SIZE);
    Model m = model_create(HIDDEN_SIZE);

    printf("Using AdamW with learning rate %f, weight decay %f\n", LEARNING_RATE, WEIGHT_DECAY);
    AdamWConfig adamw_conf = optim_adamw_get_config(LEARNING_RATE, WEIGHT_DECAY);

    arena_allocator* epoch_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(epoch_arena);

    printf("Training for %u epochs\n", EPOCHS);

    for (u32 i = 0; i < EPOCHS; i++) {
        f32 train_loss = 0, train_acc = 0;
        for (u32 j = 0; j < train_batches_size; j++) {
            GradTensor* pred = model_forward(&m, train_batches[j]);
            GradTensor* loss = nn_cross_enropy_loss(pred, train_labels[j]);
            gradt_backward(loss, optim_adamw, &adamw_conf);
            optim_adamw_step(&adamw_conf);

            train_loss += loss->tens->data[0];
            train_acc += gradt_compute_accuracy(pred, train_labels[j]);

            arena_free(epoch_arena);
        }

        train_loss /= train_batches_size;
        train_acc /= train_batches_size;

        f32 val_loss = 0, val_acc = 0;
        for (u32 j = 0; j < val_batches_size; j++) {
            GradTensor* pred = model_forward(&m, val_batches[j]);
            GradTensor* loss = nn_cross_enropy_loss(pred, val_labels[j]);

            val_loss += loss->tens->data[0];
            val_acc += gradt_compute_accuracy(pred, val_labels[j]);

            arena_free(epoch_arena);
        }

        val_loss /= val_batches_size;
        val_acc /= val_batches_size;
        printf("Epoch %u: train loss %f, accuracy %f | validation loss %f, accuracy %f\n", i, train_loss, train_acc, val_loss, val_acc);
    }

    f32 test_loss = 0, test_acc = 0;
    for (u32 j = 0; j < test_batches_size; j++) {
        GradTensor* pred = model_forward(&m, test_batches[j]);
        GradTensor* loss = nn_cross_enropy_loss(pred, test_labels[j]);

        test_loss += loss->tens->data[0];
        test_acc += gradt_compute_accuracy(pred, test_labels[j]);

        arena_free(epoch_arena);
    }
    test_loss /= test_batches_size;
    test_acc /= test_batches_size;
    printf("Test loss: %f, accuracy %f\n", test_loss, test_acc);
    
    arena_destroy(epoch_arena);
    arena_destroy(permanent_arena);
}

static IrisClass parse_species(const char* species) {
    if (strncmp(species, "Iris-setosa", 11) == 0)     return SETOSA;
    if (strncmp(species, "Iris-versicolor", 15) == 0)  return VERSICOLOR;
    if (strncmp(species, "Iris-virginica", 14) == 0)   return VIRGINICA;
    fprintf(stderr, "Unknown species: %s\n", species);
    exit(1);
}

static u32 count_lines(FILE* f) {
    u32 count = 0;
    char line[IRIS_MAX_LINE];
    while (fgets(line, sizeof(line), f)) count++;
    rewind(f);
    return count;
}

static IrisEntry* load_iris(arena_allocator* arena, const char* path, u32* out_count) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }

    u32 n_lines = count_lines(f);
    u32 n_entries = n_lines > 0 ? n_lines - 1 : 0; // exclude header

    IrisEntry* entries = arena_alloc(arena, sizeof(IrisEntry), n_entries);
    char line[IRIS_MAX_LINE];

    // skip header
    fgets(line, sizeof(line), f);

    u32 count = 0;
    while (fgets(line, sizeof(line), f) && count < n_entries) {
        // strip newline
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;

        char* tok = strtok(line, ",");
        entries[count].sepal_length = strtof(tok, NULL);
        tok = strtok(NULL, ",");
        entries[count].sepal_width = strtof(tok, NULL);
        tok = strtok(NULL, ",");
        entries[count].petal_length = strtof(tok, NULL);
        tok = strtok(NULL, ",");
        entries[count].petal_width = strtof(tok, NULL);
        tok = strtok(NULL, ",");
        entries[count].class = parse_species(tok);

        count++;
    }

    fclose(f);
    *out_count = count;
    return entries;
}

static IrisEntry* shuffle_iris(arena_allocator* arena, IrisEntry* entries, u32 count) {
    IrisEntry* shuffled = arena_alloc(arena, sizeof(IrisEntry), count);
    memcpy(shuffled, entries, sizeof(IrisEntry) * count);

    // Fisher-Yates shuffle
    for (u32 i = count - 1; i > 0; i--) {
        u32 j = rand() % (i + 1);
        IrisEntry tmp = shuffled[i];
        shuffled[i] = shuffled[j];
        shuffled[j] = tmp;
    }

    return shuffled;
}

static GradTensor** create_batches(IrisEntry* entries, u32 count, u32* out_n_batches, arena_allocator* arena) {
    u32 n_batches = (count + BATCH_SIZE - 1) / BATCH_SIZE;
    GradTensor** batches = arena_alloc(arena, sizeof(GradTensor*), n_batches);

    for (u32 b = 0; b < n_batches; b++) {
        u32 batch_len = (b < n_batches - 1) ? BATCH_SIZE : count - b * BATCH_SIZE;
        u32 shape[] = {1, 1, batch_len, 4};
        batches[b] = gradt_create_nograd(shape, 4);

        f32 buffer[batch_len * 4];
        for (u32 i = 0; i < batch_len; i++) {
            IrisEntry* e = &entries[b * BATCH_SIZE + i];
            buffer[i * 4 + 0] = e->sepal_length;
            buffer[i * 4 + 1] = e->sepal_width;
            buffer[i * 4 + 2] = e->petal_length;
            buffer[i * 4 + 3] = e->petal_width;
        }
        tensor_set_buffer(batches[b]->tens, buffer, batch_len * 4);
    }

    *out_n_batches = n_batches;
    return batches;
}

static GradTensor** create_label_batches(IrisEntry* entries, u32 count, u32* out_n_batches, arena_allocator* arena) {
    u32 n_batches = (count + BATCH_SIZE - 1) / BATCH_SIZE;
    GradTensor** batches = arena_alloc(arena, sizeof(GradTensor*), n_batches);

    for (u32 b = 0; b < n_batches; b++) {
        u32 batch_len = (b < n_batches - 1) ? BATCH_SIZE : count - b * BATCH_SIZE;
        u32 shape[] = {1, 1, batch_len, 3};
        batches[b] = gradt_create_nograd(shape, 4);

        f32 buffer[batch_len * 3];
        memset(buffer, 0, sizeof(f32) * batch_len * 3);
        for (u32 i = 0; i < batch_len; i++) {
            IrisEntry* e = &entries[b * BATCH_SIZE + i];
            buffer[i * 3 + e->class] = 1.0f;
        }
        tensor_set_buffer(batches[b]->tens, buffer, batch_len * 3);
    }

    *out_n_batches = n_batches;
    return batches;
}

static Model model_create(u32 hidden_size) {
    Model m = {
        .h1 = nn_linear_create(4, hidden_size),
        .h2 = nn_linear_create(hidden_size, hidden_size),
        .output = nn_linear_create(hidden_size, 3)
    };
    return m;
}

static GradTensor* model_forward(Model* m, GradTensor* x) {
    GradTensor* out = nn_linear_forward(&m->h1, x);
    out = nn_relu(out);
    out = nn_linear_forward(&m->h2, out);
    out = nn_relu(out);
    out = nn_linear_forward(&m->output, out);
    return out;
}
