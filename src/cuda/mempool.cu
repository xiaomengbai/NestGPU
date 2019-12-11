#include "../include/mempool.h"

char * __mp_base__;
char * __mp_free__;

gpu_mempool gpu_inner_mp = { 0 };
gpu_mempool gpu_inter_mp = { 0 };
