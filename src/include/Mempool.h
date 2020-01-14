#ifndef __M_MEMPOOL_H__
#define __M_MEMPOOL_H__
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include "gpuCudaLib.h"
#include "common.h"


class Mempool {
public:
    static const size_t CPU_INIT_SIZE = 128 * 1024 * 1024;
    static const size_t GPU_INIT_SIZE = 128 * 1024 * 1024;

    Mempool(int _type);
    virtual ~Mempool();

    char *freepos() const { return free; }
    void freeto(char *pos);
    void freeall() { free = base; }
    size_t usedsize() const { return free - base; }
    size_t freesize() const { return size - (free - base); }
    size_t getsize() const { return size; }

    void resize(size_t newsize);

    char *alloc(size_t _size);
private:

    char * base;
    char * free;
    size_t size;
    int    type;
};

#endif
