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

arena_allocator* create_arena(usize reserve_size, usize commit_size, usize alignment);
bool destroy_arena(arena_allocator* arena);

void* alloc_arena(arena_allocator* arena, usize el_size, usize n_el);
void free_arena(arena_allocator* arena);
void free_size_arena(arena_allocator* arena, usize size);
void free_to_arena(arena_allocator* arena, usize new_pos);


#endif
