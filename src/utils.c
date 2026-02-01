#include "../include/utils.h"

#include <stdlib.h>
#include <time.h>

void init_random() {
    srand(time(NULL));
}

f32 random_f32(f32 min, f32 max) {
    f32 scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

u64 perf_counter_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)(ts.tv_sec) * 1000000000 + (uint64_t)ts.tv_nsec;
}

DynArray create_dynarr(usize cap) {
    void** ptr = malloc(cap * sizeof(void*));
    DynArray a = {.len = 0, .cap = cap, .ptr = ptr};
    return a;
}

void push_dynarr(DynArray* a, void* el){
    if (a->len >= a->cap) {
        a->cap *= 1.5;
        a->ptr = realloc(a->ptr, a->cap);
    }
    a->ptr[a->len] = el;
    a->len++;
}

bool contains(DynArray* a, void* el) {
    for (usize i = 0; i < a->len; i++) {
        if (a->ptr[i] == el) {
            return true;
        }
    }
    return false;
}
void free_dynarr(DynArray* a) {
    free(a->ptr);
}
