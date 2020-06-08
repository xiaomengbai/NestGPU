#include "../include/Mempool.h"
#include <stdint.h>
Mempool::Mempool(int _type) :
    type(_type)
{
    if(type == MEM) {
        size = CPU_INIT_SIZE;
        base = (char *)malloc(size);
        assert(base != NULL);
    }else if(type == GPU) {
        size = GPU_INIT_SIZE;
        CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&base, size) );
    }
    free = base;
}

Mempool::~Mempool()
{
    if(type == MEM) {
        ::free(base);
    } else if(type == GPU) {
        CUDA_SAFE_CALL_NO_SYNC( cudaFree(base) );
    }
}

void Mempool::freeto(char *pos)
{
    assert(pos <= free && pos >= base);
    free = pos;
}

void Mempool::resize(size_t newsize)
{
    if(newsize <= size)
        return;

    if(type == MEM) {
        size_t used = usedsize();
        base = (char *) realloc(base, newsize);
        assert(base != NULL);
        free = base + used;
    }else if(type == GPU) {
        char *tmp;
        CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&tmp, newsize) );
        CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(tmp, base, free - base, cudaMemcpyDeviceToDevice) );
        free = tmp + (free - base);
        CUDA_SAFE_CALL_NO_SYNC( cudaFree(base) );
        base = tmp;
    }
    size = newsize;
}

char *Mempool::alloc(size_t _size)
{
    // while(_size > freesize())
    //     resize(size * 2);

    char *tmp = free;
    free = free + _size;
    return tmp;
}

char *Mempool::alloc_memalign(size_t _size)
{
    char *tmp;
    if ( ((uintptr_t)free & 0x7) == 0 )
        tmp = free;
    else
        tmp = (char *)(((uintptr_t)free & (~0x7)) + 0x8);
    //char *tmp = (free & 0x7 == 0) ? free : (free + ((free >> 3) + 1) << 3 );
    free = tmp + _size;
    return tmp;
}