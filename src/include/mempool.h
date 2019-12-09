#ifndef __M_MEMPOOL__
#define __M_MEMPOOL__

#include <stdlib.h>
#include <assert.h>

extern char * __mp_base__;
extern char * __mp_free__;

#define MEMPOOL_SZ (128 * 1024 * 1024)

#define MEMPOOL_CHECK() assert(__mp_free__ - __mp_base__ < MEMPOOL_SZ)

static inline void init_mempool()
{
    __mp_base__ = (char *)malloc(MEMPOOL_SZ);
    assert(__mp_base__ != NULL);
    __mp_free__ = __mp_base__;
}

static inline char *alloc_mempool(size_t size)
{
    char *tmp = __mp_free__;
    __mp_free__ = __mp_free__ + size;
    return tmp;
}

static inline void freeto_mempool(char *reset)
{
    __mp_free__ = reset;
}

static inline char *freepos_mempool()
{
    return __mp_free__;
}

static inline void destroy_mempool()
{
    free(__mp_base__);
}

#endif
