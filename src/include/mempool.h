#ifndef __M_MEMPOOL__
#define __M_MEMPOOL__

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include "gpuCudaLib.h"

extern char * __mp_base__;
extern char * __mp_free__;

typedef struct _gpu_mempool {
    char * base;
    char * free;
    size_t size;
} gpu_mempool;

extern gpu_mempool gpu_inner_mp;
extern gpu_mempool gpu_inter_mp;

#define MEMPOOL_SZ (128 * 1024 * 1024)
#define GPU_MEMPOOL_SZ MEMPOOL_SZ

#define MEMPOOL_CHECK() assert(__mp_free__ - __mp_base__ < MEMPOOL_SZ)
#define GPU_MEMPOOL_CHECK(mp) assert(mp.free - mp.base < mp.size)

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


static inline void init_gpu_mempool(gpu_mempool *pool, size_t pool_size = GPU_MEMPOOL_SZ)
{
    CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&(pool->base), pool_size) );
    pool->free = pool->base;
    pool->size = pool_size;
}

static inline void resize_gpu_mempool(gpu_mempool *pool, size_t new_size)
{
    if(new_size <= pool->size)
        return;
    char *tmp;
    CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&tmp, new_size) );
    CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(tmp, pool->base, pool->free - pool->base, cudaMemcpyDeviceToDevice) );
    pool->free = tmp + (pool->free - pool->base);
    CUDA_SAFE_CALL_NO_SYNC( cudaFree(pool->base) );
    pool->base = tmp;
    pool->size = new_size;
}

static inline void destroy_gpu_mempool(gpu_mempool *pool)
{
    CUDA_SAFE_CALL_NO_SYNC( cudaFree(pool->base) );
}

static inline void alloc_gpu_mempool(gpu_mempool *pool, char **addr, size_t size)
{
    *addr = pool->free;
    pool->free = pool->free + size;

}

static inline char *freepos_gpu_mempool(gpu_mempool *pool)
{
    return pool->base;
}

static inline void freeto_gpu_mempool(gpu_mempool *pool, char *reset)
{
    pool->free = reset;
}
#endif
