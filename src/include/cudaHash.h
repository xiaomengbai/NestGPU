#ifndef __GPU_CUDAHASH__
#define __GPU_CUDAHASH__

__device__ static unsigned int StringHash(const char* s)
{
    unsigned int hash = 0;
    int c;
 
    while((c = *s++))
    {
        hash = ((hash << 5) + hash) ^ c;
    }
 
    return hash;
}

__device__ static unsigned int StringHashInt(const char *s)
{
    unsigned int hash = 0;
    int c;

    while((c = *s++))
    {
        hash = hash == 0 ? (c - 30) : hash * 10 + (c - 30);
    }
    return hash;
}

#endif
