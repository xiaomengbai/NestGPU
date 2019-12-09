#ifndef __M_MEMPOOL__
#define __M_MEMPOOL__

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include "gpuCudaLib.h"

extern char * __mp_base__;
extern char * __mp_free__;

extern char * __gpu_mp_base__;
extern char * __gpu_mp_free__;
extern size_t __gpu_mp_size__;

#define MEMPOOL_SZ (128 * 1024 * 1024)
#define GPU_MEMPOOL_SZ MEMPOOL_SZ

#define MEMPOOL_CHECK() assert(__mp_free__ - __mp_base__ < MEMPOOL_SZ)
#define GPU_MEMPOOL_CHECK() assert(__gpu_mp_free__ - __gpu_mp_base__ < __gpu_mp_size__)

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


static inline void init_gpu_mempool(size_t pool_size = GPU_MEMPOOL_SZ)
{
    CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&__gpu_mp_base__, pool_size) );
    __gpu_mp_free__ = __gpu_mp_base__;
    __gpu_mp_size__ = pool_size;
}

static inline void resize_gpu_mempool(size_t new_size)
{
    if(new_size <= __gpu_mp_size__)
        return;
    char *tmp;
    CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&tmp, new_size) );
    CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(tmp, __gpu_mp_base__, __gpu_mp_free__ - __gpu_mp_base__, cudaMemcpyDeviceToDevice) );
    CUDA_SAFE_CALL_NO_SYNC( cudaFree(__gpu_mp_base__) );
    __gpu_mp_free__ = tmp + (__gpu_mp_free__ - __gpu_mp_base__);
    __gpu_mp_base__ = tmp;
    __gpu_mp_size__ = new_size;
}

static inline void destroy_gpu_mempool()
{
    CUDA_SAFE_CALL_NO_SYNC( cudaFree(__gpu_mp_base__) );
}

static inline void alloc_gpu_mempool(char **addr, size_t size)
{
    *addr = __gpu_mp_free__;
    __gpu_mp_free__ = __gpu_mp_free__ + size;
}

static inline char *freepos_gpu_mempool()
{
    return __gpu_mp_free__;
}

static inline void freeto_gpu_mempool(char *reset)
{
    __gpu_mp_free__ = reset;
}

#endif
