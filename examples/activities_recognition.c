#include "../include/grad.h"
#include "../include/nn.h"
#include "../include/optim.h"
#include <string.h>

#define AR_PATH "examples/datasets/activities_recognition.txt"
#define AR_MAX_LINE 256

const u32 N_VAL_USERS = 5;
const u32 N_TEST_USERS = 5;
const u32 WINDOW = 100;
const u32 STRIDE = 20;
const u32 BATCH_SIZE = 64;
const u32 N_FEATURES = 3;
const u32 N_CLASSES = 6;
const u32 HIDDEN_SIZE = 64;
const f32 LEARNING_RATE = 1e-3;
const f32 WEIGHT_DECAY = 1e-4;
const u32 EPOCHS = 50;

typedef enum: u32 {
    WALKING = 0,
    JOGGING,
    UPSTAIRS,
    DOWNSTAIRS,
    SITTING,
    STANDING
} ARClass;

typedef struct {
    u32 user_id;
    u64 timestamp;
    f32 x_axis;
    f32 y_axis;
    f32 z_axis;
    ARClass class;
} AREntry;

typedef struct {
    LSTM lstm;
    LinearLayer hidden;
    LinearLayer output;
} Model;

static ARClass parse_activity(const char* activity);
static u32 count_lines(FILE* f);
static AREntry* load_ar(arena_allocator* arena, const char* path, u32* out_count);
static void build_sequences(AREntry* entries, u32 count, arena_allocator* arena,
                            f32** out_seqs, u32** out_labels, u32* out_n_seqs);
static GradTensor** create_batches(f32* seqs, u32 n_seqs, u32* out_n_batches, arena_allocator* arena);
static GradTensor** create_label_batches(u32* labels, u32 n_seqs, u32* out_n_batches, arena_allocator* arena);
static Model model_create(u32 hidden_size);
static GradTensor* model_forward(Model* m, GradTensor* x);

int main() {
    printf("Activities recognition dataset example\n");
    arena_allocator* permanent_arena = arena_create(GiB(1), MiB(1), 8);
    gradt_set_arena(permanent_arena);

    u32 count;
    AREntry* data = load_ar(permanent_arena, AR_PATH, &count);
    printf("Loaded %u entries\n", count);

    u32 n_users = 0;
    u32 user_ids[256];
    for (u32 i = 0; i < count; i++) {
        bool found = false;
        for (u32 j = 0; j < n_users; j++) {
            if (user_ids[j] == data[i].user_id) { found = true; break; }
        }
        if (!found) user_ids[n_users++] = data[i].user_id;
    }
    printf("Found %u unique users\n", n_users);

    for (u32 i = n_users - 1; i > 0; i--) {
        u32 j = rand() % (i + 1);
        u32 tmp = user_ids[i];
        user_ids[i] = user_ids[j];
        user_ids[j] = tmp;
    }

    u32* test_users = user_ids;
    u32* val_users = &user_ids[N_TEST_USERS];

    u32 train_count = 0, val_count = 0, test_count = 0;
    for (u32 i = 0; i < count; i++) {
        u32 uid = data[i].user_id;
        for (u32 j = 0; j < N_TEST_USERS; j++)
            if (test_users[j] == uid) { test_count++; goto next; }
        for (u32 j = 0; j < N_VAL_USERS; j++)
            if (val_users[j] == uid) { val_count++; goto next; }
        train_count++;
        next:;
    }

    AREntry* train = arena_alloc(permanent_arena, sizeof(AREntry), train_count);
    AREntry* val   = arena_alloc(permanent_arena, sizeof(AREntry), val_count);
    AREntry* test  = arena_alloc(permanent_arena, sizeof(AREntry), test_count);

    u32 ti = 0, vi = 0, tei = 0;
    for (u32 i = 0; i < count; i++) {
        u32 uid = data[i].user_id;
        for (u32 j = 0; j < N_TEST_USERS; j++)
            if (test_users[j] == uid) { test[tei++] = data[i]; goto next2; }
        for (u32 j = 0; j < N_VAL_USERS; j++)
            if (val_users[j] == uid) { val[vi++] = data[i]; goto next2; }
        train[ti++] = data[i];
        next2:;
    }

    printf("Train: %u, Validation: %u, Test: %u\n", train_count, val_count, test_count);

    // build sequences
    f32* train_seqs; u32* train_labels; u32 n_train_seqs;
    build_sequences(train, train_count, permanent_arena, &train_seqs, &train_labels, &n_train_seqs);
    printf("Train sequences: %u\n", n_train_seqs);

    f32* val_seqs; u32* val_labels; u32 n_val_seqs;
    build_sequences(val, val_count, permanent_arena, &val_seqs, &val_labels, &n_val_seqs);
    printf("Validation sequences: %u\n", n_val_seqs);

    f32* test_seqs; u32* test_labels; u32 n_test_seqs;
    build_sequences(test, test_count, permanent_arena, &test_seqs, &test_labels, &n_test_seqs);
    printf("Test sequences: %u\n", n_test_seqs);

    // create batches
    u32 train_batches_size;
    GradTensor** train_batches = create_batches(train_seqs, n_train_seqs, &train_batches_size, permanent_arena);
    u32 train_label_batches_size;
    GradTensor** train_label_b = create_label_batches(train_labels, n_train_seqs, &train_label_batches_size, permanent_arena);
    printf("Train batches: %u, Label batches: %u\n", train_batches_size, train_label_batches_size);

    u32 val_batches_size;
    GradTensor** val_batches = create_batches(val_seqs, n_val_seqs, &val_batches_size, permanent_arena);
    u32 val_label_batches_size;
    GradTensor** val_label_b = create_label_batches(val_labels, n_val_seqs, &val_label_batches_size, permanent_arena);
    printf("Validation batches: %u, Label batches: %u\n", val_batches_size, val_label_batches_size);

    u32 test_batches_size;
    GradTensor** test_batches = create_batches(test_seqs, n_test_seqs, &test_batches_size, permanent_arena);
    u32 test_label_batches_size;
    GradTensor** test_label_b = create_label_batches(test_labels, n_test_seqs, &test_label_batches_size, permanent_arena);
    printf("Test batches: %u, Label batches: %u\n", test_batches_size, test_label_batches_size);

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
            GradTensor* loss = nn_cross_enropy_loss(pred, train_label_b[j]);
            gradt_backward(loss, optim_adamw, &adamw_conf);

            train_loss += loss->tens->data[0];
            train_acc += gradt_compute_accuracy(pred, train_label_b[j]);

            arena_free(epoch_arena);
        }
        train_loss /= train_batches_size;
        train_acc /= train_batches_size;

        f32 val_loss = 0, val_acc = 0;
        for (u32 j = 0; j < val_batches_size; j++) {
            GradTensor* pred = model_forward(&m, val_batches[j]);
            GradTensor* loss = nn_cross_enropy_loss(pred, val_label_b[j]);

            val_loss += loss->tens->data[0];
            val_acc += gradt_compute_accuracy(pred, val_label_b[j]);

            arena_free(epoch_arena);
        }

        val_loss /= val_batches_size;
        val_acc /= val_batches_size;
        printf("Epoch %u: train loss %f, accuracy %f | validation loss %f, accuracy %f\n", i, train_loss, train_acc, val_loss, val_acc);
    }

    f32 test_loss = 0, test_acc = 0;
    for (u32 j = 0; j < test_batches_size; j++) {
        GradTensor* pred = model_forward(&m, test_batches[j]);
        GradTensor* loss = nn_cross_enropy_loss(pred, test_label_b[j]);

        test_loss += loss->tens->data[0];
        test_acc += gradt_compute_accuracy(pred, test_label_b[j]);

        arena_free(epoch_arena);
    }
    test_loss /= test_batches_size;
    test_acc /= test_batches_size;
    printf("Test loss: %f, accuracy %f\n", test_loss, test_acc);
    
    arena_destroy(epoch_arena);
    arena_destroy(permanent_arena);
}

static ARClass parse_activity(const char* activity) {
    if (strncmp(activity, "Walking", 7) == 0)    return WALKING;
    if (strncmp(activity, "Jogging", 7) == 0)    return JOGGING;
    if (strncmp(activity, "Upstairs", 8) == 0)   return UPSTAIRS;
    if (strncmp(activity, "Downstairs", 10) == 0) return DOWNSTAIRS;
    if (strncmp(activity, "Sitting", 7) == 0)    return SITTING;
    if (strncmp(activity, "Standing", 8) == 0)   return STANDING;
    fprintf(stderr, "Unknown activity: %s\n", activity);
    exit(1);
}

static u32 count_lines(FILE* f) {
    u32 count = 0;
    char line[AR_MAX_LINE];
    while (fgets(line, sizeof(line), f)) count++;
    rewind(f);
    return count;
}

static AREntry* load_ar(arena_allocator* arena, const char* path, u32* out_count) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }

    u32 n_lines = count_lines(f);

    AREntry* entries = arena_alloc(arena, sizeof(AREntry), n_lines);
    char line[AR_MAX_LINE];

    u32 count = 0;
    while (fgets(line, sizeof(line), f) && count < n_lines) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;

        char* tok = strtok(line, ",");
        if (!tok) continue;
        entries[count].user_id = (u32)strtoul(tok, NULL, 10);
        tok = strtok(NULL, ",");
        if (!tok) continue;
        entries[count].class = parse_activity(tok);
        tok = strtok(NULL, ",");
        if (!tok) continue;
        entries[count].timestamp = (u64)strtoull(tok, NULL, 10);
        tok = strtok(NULL, ",");
        if (!tok) continue;
        entries[count].x_axis = strtof(tok, NULL);
        tok = strtok(NULL, ",");
        if (!tok) continue;
        entries[count].y_axis = strtof(tok, NULL);
        tok = strtok(NULL, ",");
        if (!tok) continue;
        entries[count].z_axis = strtof(tok, NULL);

        count++;
    }

    fclose(f);
    *out_count = count;
    return entries;
}

static void build_sequences(AREntry* entries, u32 count, arena_allocator* arena,
                            f32** out_seqs, u32** out_labels, u32* out_n_seqs) {
    // first pass: count total sequences
    u32 total_seqs = 0;
    u32 seg_start = 0;
    while (seg_start < count) {
        // find end of segment (consecutive entries with same user_id and class)
        u32 seg_end = seg_start + 1;
        while (seg_end < count &&
               entries[seg_end].user_id == entries[seg_start].user_id &&
               entries[seg_end].class == entries[seg_start].class) {
            seg_end++;
        }
        u32 seg_len = seg_end - seg_start;
        u32 padded_len = seg_len + (WINDOW - seg_len % WINDOW) % WINDOW;
        u32 n_windows = (padded_len - WINDOW) / STRIDE + 1;
        total_seqs += n_windows;
        seg_start = seg_end;
    }

    // allocate output arrays
    f32* seqs = arena_alloc(arena, sizeof(f32), total_seqs * WINDOW * N_FEATURES);
    u32* labels = arena_alloc(arena, sizeof(u32), total_seqs);

    // second pass: fill sequences
    u32 seq_idx = 0;
    seg_start = 0;
    while (seg_start < count) {
        u32 seg_end = seg_start + 1;
        while (seg_end < count &&
               entries[seg_end].user_id == entries[seg_start].user_id &&
               entries[seg_end].class == entries[seg_start].class) {
            seg_end++;
        }
        u32 seg_len = seg_end - seg_start;
        u32 padded_len = seg_len + (WINDOW - seg_len % WINDOW) % WINDOW;
        ARClass label = entries[seg_start].class;

        u32 idx = 0;
        while (idx + WINDOW <= padded_len) {
            f32* dst = &seqs[seq_idx * WINDOW * N_FEATURES];
            for (u32 t = 0; t < WINDOW; t++) {
                u32 src_idx = seg_start + idx + t;
                if (src_idx < seg_end) {
                    dst[t * N_FEATURES + 0] = entries[src_idx].x_axis;
                    dst[t * N_FEATURES + 1] = entries[src_idx].y_axis;
                    dst[t * N_FEATURES + 2] = entries[src_idx].z_axis;
                } else {
                    dst[t * N_FEATURES + 0] = 0.0f;
                    dst[t * N_FEATURES + 1] = 0.0f;
                    dst[t * N_FEATURES + 2] = 0.0f;
                }
            }
            labels[seq_idx] = label;
            seq_idx++;
            idx += STRIDE;
        }

        seg_start = seg_end;
    }

    *out_seqs = seqs;
    *out_labels = labels;
    *out_n_seqs = total_seqs;
}

static GradTensor** create_batches(f32* seqs, u32 n_seqs, u32* out_n_batches, arena_allocator* arena) {
    u32 n_batches = (n_seqs + BATCH_SIZE - 1) / BATCH_SIZE;
    GradTensor** batches = arena_alloc(arena, sizeof(GradTensor*), n_batches);

    for (u32 b = 0; b < n_batches; b++) {
        u32 batch_len = (b < n_batches - 1) ? BATCH_SIZE : n_seqs - b * BATCH_SIZE;
        u32 shape[] = {1, WINDOW, batch_len, N_FEATURES};
        batches[b] = gradt_create_nograd(shape, 4);

        f32 buffer[WINDOW * batch_len * N_FEATURES];
        for (u32 t = 0; t < WINDOW; t++) {
            for (u32 s = 0; s < batch_len; s++) {
                f32* src = &seqs[(b * BATCH_SIZE + s) * WINDOW * N_FEATURES + t * N_FEATURES];
                f32* dst = &buffer[(t * batch_len + s) * N_FEATURES];
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
            }
        }
        tensor_set_buffer(batches[b]->tens, buffer, WINDOW * batch_len * N_FEATURES);
    }

    *out_n_batches = n_batches;
    return batches;
}

static GradTensor** create_label_batches(u32* labels, u32 n_seqs, u32* out_n_batches, arena_allocator* arena) {
    u32 n_batches = (n_seqs + BATCH_SIZE - 1) / BATCH_SIZE;
    GradTensor** batches = arena_alloc(arena, sizeof(GradTensor*), n_batches);

    for (u32 b = 0; b < n_batches; b++) {
        u32 batch_len = (b < n_batches - 1) ? BATCH_SIZE : n_seqs - b * BATCH_SIZE;
        u32 shape[] = {1, 1, batch_len, N_CLASSES};
        batches[b] = gradt_create_nograd(shape, 4);

        f32 buffer[batch_len * N_CLASSES];
        memset(buffer, 0, sizeof(f32) * batch_len * N_CLASSES);
        for (u32 i = 0; i < batch_len; i++) {
            u32 label = labels[b * BATCH_SIZE + i];
            buffer[i * N_CLASSES + label] = 1.0f;
        }
        tensor_set_buffer(batches[b]->tens, buffer, batch_len * N_CLASSES);
    }

    *out_n_batches = n_batches;
    return batches;
}

static Model model_create(u32 hidden_size) {
    Model m = {
        .lstm = nn_lstm_init(N_FEATURES, hidden_size),
        .hidden = nn_linear_create(hidden_size, hidden_size),
        .output = nn_linear_create(hidden_size, N_CLASSES)
    };
    return m;
}

static GradTensor* model_forward(Model* m, GradTensor* x) {
    GradTensor* t = nn_lstm_forward(&m->lstm, x);
    t = nn_linear_forward(&m->hidden, t);
    t = nn_relu(t);
    t = nn_linear_forward(&m->output, t);
    return t;
}
