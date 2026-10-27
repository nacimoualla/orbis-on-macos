/*
 * libc routines that call back into guest code, run AS guest code.
 *
 * bsearch/qsort take a guest comparator. HLE handlers never re-enter the
 * guest CPU, so instead of calling the comparator from the host these are
 * compiled to x86-64 and copied into guest memory; the HLE handler for the
 * NID tail-jumps here and the routine's `ret` returns straight to the caller.
 *
 * Constraints: position independent, no calls except through the comparator
 * pointer, no data sections, integer-only (no SSE/AVX), so the bytes can be
 * dropped anywhere and run on the interpreter and the JIT.
 *
 * Regenerate include/ps4/guest_libc_blob.hpp after editing:
 *     python3 scripts/gen_guest_libc.py
 */
typedef unsigned long size_t;
typedef int (*cmp_fn)(const void*, const void*);

/* Entry points are at fixed offsets: keep guest_bsearch first. */
void* guest_bsearch(const void* key, const void* base, size_t n, size_t size, cmp_fn cmp) {
    const char* lo = (const char*)base;
    while (n > 0) {
        const char* mid = lo + (n / 2) * size;
        const int c = cmp(key, mid);
        if (c == 0) return (void*)mid;
        if (c > 0) {
            lo = mid + size;
            n = n - n / 2 - 1;
        } else {
            n = n / 2;
        }
    }
    return 0;
}

static void swap_bytes(char* a, char* b, size_t size) {
    while (size--) {
        const char t = *a;
        *a++ = *b;
        *b++ = t;
    }
}

static void sift_down(char* base, size_t root, size_t n, size_t size, cmp_fn cmp) {
    for (;;) {
        size_t child = 2 * root + 1;
        if (child >= n) return;
        if (child + 1 < n && cmp(base + child * size, base + (child + 1) * size) < 0) ++child;
        if (cmp(base + root * size, base + child * size) >= 0) return;
        swap_bytes(base + root * size, base + child * size, size);
        root = child;
    }
}

/* Heapsort: O(n log n), in place, no recursion or scratch memory. */
void guest_qsort(void* vbase, size_t n, size_t size, cmp_fn cmp) {
    char* base = (char*)vbase;
    if (n < 2 || size == 0) return;
    for (size_t i = n / 2; i-- > 0;) sift_down(base, i, n, size, cmp);
    for (size_t end = n - 1; end > 0; --end) {
        swap_bytes(base, base + end * size, size);
        sift_down(base, 0, end, size, cmp);
    }
}
