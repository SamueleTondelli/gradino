#ifndef ARENA_H
#define ARENA_H

#include <stdbool.h>

#include "utils.h"

typedef struct {
    usize reserve_size;
    usize commit_size;
    usize alloc_pos;
    usize commit_pos;
    usize alignment;
} arena_allocator;

arena_allocator* arena_create(usize reserve_size, usize commit_size, usize alignment);
bool arena_destroy(arena_allocator* arena);

void* arena_alloc(arena_allocator* arena, usize el_size, usize n_el);
void arena_free(arena_allocator* arena);
void arena_free_size(arena_allocator* arena, usize size);
void arena_free_to(arena_allocator* arena, usize new_pos);


#endif
