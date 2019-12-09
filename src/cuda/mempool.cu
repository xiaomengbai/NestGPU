#include "../include/mempool.h"

char * __mp_base__;
char * __mp_free__;

char * __gpu_mp_base__;
char * __gpu_mp_free__;
size_t __gpu_mp_size__ = 0;