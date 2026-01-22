#include "../include/arena.h"

#include <stdalign.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/mman.h>

#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

static void* reserve_mem(usize reserve_size) {
    void* mem = mmap(NULL, reserve_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        return NULL;
    }
    return mem;
}

static bool commit_mem(void* mem, usize commit_size) {
    return mprotect(mem, commit_size, PROT_READ | PROT_WRITE) == 0;
}

static bool release_mem(void* mem, usize release_size) {
    return munmap(mem, release_size) == 0;
}

static u32 get_page_size() {
    return (u32)sysconf(_SC_PAGESIZE);
}

arena_allocator* create_arena(usize reserve_size, usize commit_size, usize alignment) {
    u32 page_size = get_page_size();

    usize reserve_aligned = ALIGN_UP_POW2(reserve_size, page_size);
    usize commit_aligned = ALIGN_UP_POW2(commit_size, page_size);
    void* mem = reserve_mem(reserve_aligned);
    if (mem == NULL) {
        return NULL;
    }

    if (!commit_mem(mem, commit_aligned)) {
        return NULL;
    }

    arena_allocator* ret = (arena_allocator*)mem;
    ret->reserve_size = reserve_aligned;
    ret->commit_size = commit_aligned;
    ret->alloc_pos = ALIGN_UP_POW2(sizeof(arena_allocator), alignment);
    ret->commit_pos = commit_aligned;
    ret->alignment = alignment;
    return ret;
}

bool destroy_arena(arena_allocator* arena) {
    return release_mem(arena, arena->reserve_size);
}

void* alloc_arena(arena_allocator* arena, usize el_size, usize n_el) {
    usize size = ALIGN_UP_POW2(el_size * n_el, arena->alignment);

    if (arena->alloc_pos + size > arena->reserve_size) {
        return NULL;
    }

    void* base = arena;

    if (arena->alloc_pos + size > arena->commit_pos) {
        usize to_commit = arena->alloc_pos + size - arena->commit_pos;
        to_commit = ALIGN_UP_POW2(to_commit, arena->commit_size);
        to_commit = to_commit > arena->reserve_size ? arena->reserve_size : to_commit;
        if (!commit_mem(base, arena->commit_pos + to_commit)) return NULL;
        arena->commit_pos += to_commit;
    }

    void* mem = base + arena->alloc_pos;
    arena->alloc_pos += size;
    return mem;
}

void free_arena(arena_allocator* arena) {
    arena->alloc_pos = ALIGN_UP_POW2(sizeof(arena_allocator), arena->alignment);    
}

void free_size_arena(arena_allocator* arena, usize size) {
    usize base_pos = ALIGN_UP_POW2(sizeof(arena_allocator), arena->alignment);
    usize new_pos = size < arena->alloc_pos - base_pos ? arena->alloc_pos - size : base_pos;
    arena->alloc_pos =new_pos;
}

void free_to_arena(arena_allocator* arena, usize new_pos) {    
    usize base_pos = ALIGN_UP_POW2(sizeof(arena_allocator), arena->alignment);
    arena->alloc_pos = new_pos > base_pos ? new_pos : base_pos;
}
