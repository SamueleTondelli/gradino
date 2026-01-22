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
