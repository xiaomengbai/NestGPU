/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include "../include/common.h"
#include "../include/hashJoin.h"
#include "../include/gpuCudaLib.h"
#include "scanImpl.cu"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "../include/Mempool.h"

/*
 * Count the number of dimension keys for each bucket.
 */

__global__ static void count_hash_num(char *dim, long  inNum,int *num,int hsize){
    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *)dim)[i];
        int hKey = joinKey & (hsize-1);
        atomicAdd(&(num[hKey]),1);
    }
}

/*
 * All the buckets are stored in a continuos memory region.
 * The starting position of each bucket is stored in the psum array.
 * For star schema quereis, the size of fact table is usually much
 * larger than the size of the dimension table. In this case, hash probing is much more
 * time consuming than building hash table. By avoiding pointer, the efficiency of hash probing
 * can be improved.
 */

__global__ static void build_hash_table(char *dim, long inNum, int *psum, char * bucket,int hsize){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *) dim)[i]; 
        int hKey = joinKey & (hsize-1);
        int pos = atomicAdd(&psum[hKey],1) * 2;
        ((int*)bucket)[pos] = joinKey;
        pos += 1;
        int dimId = i+1;
        ((int*)bucket)[pos] = dimId;
    }

}

/*
 * Count join result for each thread for dictionary encoded column. 
 */

__global__ static void count_join_result_dict(int *num, int* psum, char* bucket, struct dictHeader *dheader, int dNum, int* dictFilter,int hsize){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<dNum;i+=stride){
        int fkey = dheader->hash[i];
        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int fvalue = 0;

        for(int j=0;j<keyNum;j++){
            int pSum = psum[hkey];
            int dimKey = ((int *)(bucket))[2*j + 2*pSum];
            if( dimKey == fkey){
                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                fvalue = dimId;
                break;
            }
        }

        dictFilter[i] = fvalue;
    }

}

/*
 * Transform the dictionary filter to the final filter than can be used to generate the result
 */

__global__ static void transform_dict_filter(int * dictFilter, char *fact, long tupleNum, int dNum,  int * filter){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    struct dictHeader *dheader;
    dheader = (struct dictHeader *) fact;

    int byteNum = dheader->bitNum/8;
    int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int)  ; 

    for(long i=offset; i<numInt; i += stride){
        int tmp = ((int *)(fact + sizeof(struct dictHeader)))[i];

        for(int j=0; j< sizeof(int)/byteNum; j++){
            int fkey = 0;
            memcpy(&fkey, ((char *)&tmp) + j*byteNum, byteNum);

            filter[i* sizeof(int)/byteNum + j] = dictFilter[fkey];
        }
    }
}

/*
 * count the number that is not zero in the filter
 */
__global__ static void filter_count(long tupleNum, int * count, int * factFilter){

    int lcount = 0;
    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(long i=offset; i<tupleNum; i+=stride){
        if(factFilter[i] !=0)
            lcount ++;
    }
    count[offset] = lcount;
}

/*
 * count join result for rle-compressed key.
 */

__global__ static void count_join_result_rle(int* num, int* psum, char* bucket, char* fact, long tupleNum, int * factFilter,int hsize){

    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    struct rleHeader *rheader = (struct rleHeader *)fact;
    int dNum = rheader->dictNum;

    for(int i=offset; i<dNum; i += stride){
        int fkey = ((int *)(fact+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int pSum = psum[hkey];

        for(int j=0;j<keyNum;j++){

            int dimKey = ((int *)(bucket))[2*j + 2*pSum];

            if( dimKey == fkey){

                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                for(int k=0;k<fcount;k++)
                    factFilter[fpos+k] = dimId;

                break;
            }
        }
    }

}

/*
 * Count join result for uncompressed column
 */

__global__ static void count_join_result_old(int* num, int* psum, char* bucket, char* fact, long inNum, int* count, int * factFilter,int hsize){
    int lcount = 0;
    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int fkey = ((int *)(fact))[i];
        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int fvalue = 0;

        for(int j=0;j<keyNum;j++){
            int pSum = psum[hkey];
            int dimKey = ((int *)(bucket))[2*j + 2*pSum];
            if( dimKey == fkey){
                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                lcount ++;
                fvalue = dimId;
                break;
            }
        }
        factFilter[i] = fvalue;
    }

    count[offset] = lcount;
}

__global__ static void count_join_result(int* num, int* psum, char* bucket, char* fact, long inNum, int* count, int * filterNum,int hsize){
    int lcount = 0;
    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int fkey = ((int *)(fact))[i];
       int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int fvalue = 0;

        for(int j=0;j<keyNum;j++){
            int pSum = psum[hkey];
            int dimKey = ((int *)(bucket))[2*j + 2*pSum];
            if( dimKey == fkey){
                lcount ++;
                fvalue ++;
            }
        }
        filterNum[i] = fvalue;
    }

    count[offset] = lcount;
}

__global__ static void probe_join_result(int* num, int* psum, char* bucket, char* fact, long inNum, int* filterPsum, int * JRes,int hsize){

    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int fkey = ((int *)(fact))[i];
        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int pos = filterPsum[i];

        for(int j=0;j<keyNum;j++){
            int pSum = psum[hkey];
            int dimKey = ((int *)(bucket))[2*j + 2*pSum];
            if( dimKey == fkey){
                int dimId = ((int *)(bucket))[2*j + 2*pSum + 1];
                JRes[pos ++] = dimId;
            }
        }
    }
}

/*
 * Unpact the rle-compressed data
 */

__global__ void static unpack_rle(char * fact, char * rle, long tupleNum,int dNum){

    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<dNum; i+=stride){

        int fvalue = ((int *)(fact+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

        for(int k=0;k<fcount;k++){
            ((int*)rle)[fpos + k] = fvalue;
        }
    }
}

/*
 * generate psum for RLE compressed column based on filter
 * current implementaton: scan through rle element and find the correponsding element in the filter
 */

__global__ void static rle_psum(int *count, char * fact,  long  tupleNum, int * filter){

    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    struct rleHeader *rheader = (struct rleHeader *) fact;
    int dNum = rheader->dictNum;

    for(int i= offset; i<dNum; i+= stride){

        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];
        int lcount= 0;

        for(int k=0;k<fcount;k++){
            if(filter[fpos + k]!=0)
                lcount++;
        }
        count[i] = lcount;
    }

}

/*
 * filter the column that is compressed using Run Length Encoding
 */

__global__ void static joinFact_rle(int *resPsum, char * fact,  int attrSize, long  tupleNum, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    struct rleHeader *rheader = (struct rleHeader *) fact;
    int dNum = rheader->dictNum;

    for(int i = startIndex; i<dNum; i += stride){
        int fkey = ((int *)(fact+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

        int toffset = resPsum[i];
        for(int j=0;j<fcount;j++){
            if(filter[fpos-j] !=0){
                ((int*)result)[toffset] = fkey ;
                toffset ++;
            }
        }
    }

}

/*
 * filter the column in the fact table that is compressed using dictionary encoding
 */
__global__ void static joinFact_dict_other(int *resPsum, char * fact,  struct dictHeader *dheader, int byteNum,int attrSize, long  num, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            int key = 0;
            memcpy(&key, fact + sizeof(struct dictHeader) + i* byteNum, byteNum);
            memcpy(result + localOffset, &dheader->hash[key], attrSize);
            localOffset += attrSize;
        }
    }
}

__global__ void static joinFact_dict_int(int *resPsum, char * fact, struct dictHeader *dheader, int byteNum, int attrSize, long  num, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            int key = 0;
            memcpy(&key, fact + sizeof(struct dictHeader) + i* byteNum, byteNum);
            ((int*)result)[localCount] = dheader->hash[key];
            localCount ++;
        }
    }
}

__global__ void static joinFact_other(int *resPsum, char * fact,  int attrSize, long  num, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            memcpy(result + localOffset, fact + i*attrSize, attrSize);
            localOffset += attrSize;
        }
    }
}

__global__ void static joinFact_int_old(int *resPsum, char * fact,  int attrSize, long  num, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for(long i=startIndex;i<num;i+=stride){
        if(filter[i] != 0){
            ((int*)result)[localCount] = ((int *)fact)[i];
            localCount ++;
        }
    }
}

__global__ void static joinFact_int(int *resPsum, char * fact,  int attrSize, long  num, int * filterNum, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(long i=startIndex;i<num;i+=stride){
        if(filterNum[i] != 0){
            long localCount = resPsum[i];
            for(int j = 0; j < filterNum[i]; j++)
            {
                ((int*)result)[localCount] = ((int *)fact)[i];
                localCount ++;
            }
        }
    }
}

__global__ void static joinFact(int *resPsum, char *fact, int attrSize, long num, int *filterNum, char *result)
{
    int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int stride     = blockDim.x * gridDim.x;

    for(long i = startIndex; i < num; i += stride){
        if(filterNum[i]){
            long localOffset = resPsum[i] * attrSize;
            for(int j = 0; j < filterNum[i]; j++){
                memcpy(result + localOffset, fact + i * attrSize, attrSize);
                localOffset += attrSize;
            }
        }
    }
}

__global__ void static joinDim_rle(int *resPsum, char * dim, int attrSize, long tupleNum, int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    struct rleHeader *rheader = (struct rleHeader *) dim;
    int dNum = rheader->dictNum;

    for(int i = startIndex; i<tupleNum; i += stride){
        int dimId = filter[i];
        if(dimId != 0){
            for(int j=0;j<dNum;j++){
                int dkey = ((int *)(dim+sizeof(struct rleHeader)))[j];
                int dcount = ((int *)(dim+sizeof(struct rleHeader)))[j + dNum];
                int dpos = ((int *)(dim+sizeof(struct rleHeader)))[j + 2*dNum];

                if(dpos == dimId || ((dpos < dimId) && (dpos + dcount) > dimId)){
                    ((int*)result)[localCount] = dkey ;
                    localCount ++;
                    break;
                }

            }
        }
    }
}

__global__ void static joinDim_dict_other(int *resPsum, char * dim, struct dictHeader *dheader, int byteNum, int attrSize, long num,int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        if( dimId != 0){
            int key = 0;
            memcpy(&key, dim + sizeof(struct dictHeader)+(dimId-1) * byteNum, byteNum);
            memcpy(result + localOffset, &dheader->hash[key], attrSize);
            localOffset += attrSize;
        }
    }
}

__global__ void static joinDim_dict_int(int *resPsum, char * dim, struct dictHeader *dheader, int byteNum, int attrSize, long num,int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        if( dimId != 0){
            int key = 0;
            memcpy(&key, dim + sizeof(struct dictHeader)+(dimId-1) * byteNum, byteNum);
            ((int*)result)[localCount] = dheader->hash[key];
            localCount ++;
        }
    }
}

__global__ void static joinDim_int(int *resPsum, char * dim, int attrSize, long num,int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localCount = resPsum[startIndex];

    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        if( dimId != 0){
            ((int*)result)[localCount] = ((int*)dim)[dimId-1];
            localCount ++;
        }
    }
}

__global__ void static joinDim_int_new(int *resPsum, char * dim, int attrSize, long num,int * filterNum, int * JRes, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(long i=startIndex;i<num;i+=stride){
        int dimNum = filterNum[i];
        if( dimNum != 0){
            long localCount = resPsum[i];
            int dimId;
            for(int j = 0; j < dimNum; j ++)
            {
                dimId = JRes[localCount];
                ((int*)result)[localCount] = ((int*)dim)[dimId-1];
                localCount ++;
            }
        }
    }
}

__global__ void static joinDim(int *resPsum, char *dim, int attrSize, long num, int *filterNum, int *JRes, char *result)
{
    int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int stride     = blockDim.x * gridDim.x;

    for(long i = startIndex; i < num; i += stride){
        int dimNum = filterNum[i];
        if(dimNum){
            long localCount = resPsum[i];
            int dimId;
            for(int j = 0; j < dimNum; j++){
                dimId = JRes[localCount];
                memcpy(result + localCount * attrSize, dim + (dimId - 1) * attrSize, attrSize);
                localCount++;
            }
        }
    }
}

__global__ void static joinDim_other(int *resPsum, char * dim, int attrSize, long num,int * filter, char * result){

    int startIndex = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    long localOffset = resPsum[startIndex] * attrSize;

    for(long i=startIndex;i<num;i+=stride){
        int dimId = filter[i];
        if( dimId != 0){
            memcpy(result + localOffset, dim + (dimId-1)* attrSize, attrSize);
            localOffset += attrSize;
        }
    }
}


int *buildColumnHash(struct tableNode *tNode, int colId, struct statistic *pp)
{
    if(colId >= tNode->totalAttr)
        return NULL;

    int *res;
    dim3 grid(4096);
    dim3 block(256);

    long primaryKeySize = sizeof(int) * tNode->tupleNum;

    int *gpu_psum, *gpu_hashNum;
    char *gpu_bucket;
    int *gpu_psum1 = NULL;

    int hsize = tNode->tupleNum;
    NP2(hsize);

    CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&res, sizeof(int) * hsize * 2 + primaryKeySize * 2) );
    CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **)&gpu_psum1, sizeof(int) * hsize) );

    gpu_hashNum = res;
    gpu_psum = res + hsize;
    gpu_bucket = (char *)(res + hsize * 2);

    CUDA_SAFE_CALL_NO_SYNC( cudaMemset(gpu_hashNum, 0, sizeof(int)*hsize) );

    int dataPos = tNode->dataPos[colId];
    char *gpu_dim;

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void**)&gpu_dim, primaryKeySize) );
        CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(gpu_dim, tNode->content[colId], primaryKeySize, cudaMemcpyHostToDevice) );

    } else if (dataPos == GPU || dataPos == UVA){
        gpu_dim = tNode->content[colId];
    }

    count_hash_num<<<grid, block>>>(gpu_dim, tNode->tupleNum, gpu_hashNum, hsize);

    scanImpl(gpu_hashNum, hsize, gpu_psum, pp);

    CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(gpu_psum1, gpu_psum, sizeof(int) * hsize, cudaMemcpyDeviceToDevice) );

    build_hash_table<<<grid, block>>>(gpu_dim, tNode->tupleNum, gpu_psum1, gpu_bucket, hsize);

    if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED)
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_dim));

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum1));
    return res;
}

/*
 * hashJoin implements the foreign key join between a fact table and dimension table.
 *
 * Prerequisites:
 *  1. the data to be joined can be fit into GPU device memory.
 *  2. dimension table is not compressed
 *  
 * Input:
 *  jNode: contains information about the two joined tables.
 *  pp: records statistics such as kernel execution time
 *
 * Output:
 *  A new table node
 */

struct tableNode * hashJoin(struct joinNode *jNode, struct statistic *pp,
                            Mempool *host_mp = NULL, Mempool *dev_mp = NULL, Mempool *res_mp = NULL, int *rightTableHash = NULL){


    //Start total timer
    struct timespec startS0, endS0;
    clock_gettime(CLOCK_REALTIME,&startS0);

    //Start timer for Step 1 - Allocate memory for intermediate results
    struct timespec startS1, endS1;
    clock_gettime(CLOCK_REALTIME,&startS1);

    struct tableNode * res = NULL;

    char *gpu_result = NULL, *gpu_bucket = NULL, *gpu_fact = NULL, *gpu_dim = NULL;
    int *gpu_count = NULL,  *gpu_psum = NULL, *gpu_resPsum = NULL, *gpu_hashNum = NULL;

    int defaultBlock = 4096;

    dim3 grid(defaultBlock);
    dim3 block(256);

    int blockNum;
    int threadNum;

    blockNum = jNode->leftTable->tupleNum / block.x + 1;
    if(blockNum < defaultBlock)
        grid = blockNum;
    else
        grid = defaultBlock;

    if(host_mp == NULL) {
        res = (struct tableNode*) malloc(sizeof(struct tableNode));
        CHECK_POINTER(res);
    }else
        res = (struct tableNode *) host_mp->alloc(sizeof(struct tableNode));

    res->totalAttr = jNode->totalAttr;
    res->tupleSize = jNode->tupleSize;
    if(host_mp == NULL) {
        res->attrType = (int *) malloc(res->totalAttr * sizeof(int));
        CHECK_POINTER(res->attrType);
        res->attrSize = (int *) malloc(res->totalAttr * sizeof(int));
        CHECK_POINTER(res->attrSize);
        res->attrIndex = (int *) malloc(res->totalAttr * sizeof(int));
        CHECK_POINTER(res->attrIndex);
        res->attrTotalSize = (int *) malloc(res->totalAttr * sizeof(int));
        CHECK_POINTER(res->attrTotalSize);
        res->dataPos = (int *) malloc(res->totalAttr * sizeof(int));
        CHECK_POINTER(res->dataPos);
        res->dataFormat = (int *) malloc(res->totalAttr * sizeof(int));
        CHECK_POINTER(res->dataFormat);
        res->content = (char **) malloc(res->totalAttr * sizeof(char *));
        CHECK_POINTER(res->content);
    }else{
        res->attrType = (int *) host_mp->alloc(res->totalAttr * sizeof(int));
        res->attrSize = (int *) host_mp->alloc(res->totalAttr * sizeof(int));
        res->attrIndex = (int *) host_mp->alloc(res->totalAttr * sizeof(int));
        res->attrTotalSize = (int *) host_mp->alloc(res->totalAttr * sizeof(int));
        res->dataPos = (int *) host_mp->alloc(res->totalAttr * sizeof(int));
        res->dataFormat = (int *) host_mp->alloc(res->totalAttr * sizeof(int));
        res->content = (char **) host_mp->alloc(res->totalAttr * sizeof(char *));
    }

    for(int i=0;i<jNode->leftOutputAttrNum;i++){
        int pos = jNode->leftPos[i];
        res->attrType[pos] = jNode->leftOutputAttrType[i];
        int index = jNode->leftOutputIndex[i];
        res->attrSize[pos] = jNode->leftTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    for(int i=0;i<jNode->rightOutputAttrNum;i++){
        int pos = jNode->rightPos[i];
        res->attrType[pos] = jNode->rightOutputAttrType[i];
        int index = jNode->rightOutputIndex[i];
        res->attrSize[pos] = jNode->rightTable->attrSize[index];
        res->dataFormat[pos] = UNCOMPRESSED;
    }

    long primaryKeySize = sizeof(int) * jNode->rightTable->tupleNum;

    //Stop for Step 1 - Allocate memory for intermediate results
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS1);
    pp->joinProf_step1_allocateMem += (endS1.tv_sec - startS1.tv_sec)* BILLION + endS1.tv_nsec - startS1.tv_nsec;
    
    //Start timer for Step 2 - Build hashTable
    struct timespec startS2, endS2;
    clock_gettime(CLOCK_REALTIME,&startS2);

    /*
     *  build hash table on GPU
    */

    //Start timer for Step 2.1 - Allocate memory
    struct timespec startS21, endS21;
    clock_gettime(CLOCK_REALTIME,&startS21);

    int *gpu_psum1 = NULL;
    int hsize = jNode->rightTable->tupleNum;
    NP2(hsize);

    if(rightTableHash == NULL) {
        if(dev_mp == NULL)
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_hashNum,sizeof(int)*hsize));
        else
            gpu_hashNum = (int *) dev_mp->alloc(sizeof(int) * hsize);
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_hashNum,0,sizeof(int)*hsize));
    } else {
        gpu_hashNum = rightTableHash;
    }

    int dataPos = jNode->rightTable->dataPos[jNode->rightKeyIndex];
    if(rightTableHash == NULL) {
        if(dev_mp == NULL) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_psum,hsize*sizeof(int)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_bucket, 2*primaryKeySize));
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_psum1,hsize*sizeof(int)));
        } else {
            gpu_psum   = (int *)  dev_mp->alloc(hsize * sizeof(int));
            gpu_bucket = (char *) dev_mp->alloc(2 * primaryKeySize);
            gpu_psum1  = (int *)  dev_mp->alloc(hsize * sizeof(int));
        }

        if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_dim,primaryKeySize));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim,jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,cudaMemcpyHostToDevice));

        }else if (dataPos == GPU || dataPos == UVA){
            gpu_dim = jNode->rightTable->content[jNode->rightKeyIndex];
        }
    } else {
        gpu_psum = rightTableHash + hsize;
        gpu_bucket = (char *) (rightTableHash + hsize * 2);
    }
    //Stop for Step 2.1 - Allocate memory
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS21);
    pp->joinProf_step21_allocateMem += (endS21.tv_sec - startS21.tv_sec)* BILLION + endS21.tv_nsec - startS21.tv_nsec;
        
    //Start timer for Step 2.2 - Count_hash_num
    struct timespec startS22, endS22;
    clock_gettime(CLOCK_REALTIME,&startS22);

    if(rightTableHash == NULL)
        count_hash_num<<<grid,block>>>(gpu_dim,jNode->rightTable->tupleNum,gpu_hashNum,hsize);

    //Stop for Step 2.2 - Count_hash_num
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS22);
    pp->joinProf_step22_Count_hash_num += (endS22.tv_sec - startS22.tv_sec)* BILLION + endS22.tv_nsec - startS22.tv_nsec;
        
    //Start timer for Step 2.3 - scanImpl
    struct timespec startS23, endS23;
    clock_gettime(CLOCK_REALTIME,&startS23);

    if(rightTableHash == NULL)
        scanImpl(gpu_hashNum,hsize,gpu_psum, pp);

    //Stop for Step 2.3 - scanImpl
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS23);
    pp->joinProf_step23_scanImpl += (endS23.tv_sec - startS23.tv_sec)* BILLION + endS23.tv_nsec - startS23.tv_nsec;
            
    //Start timer for Step 2.4 - build_hash_table (+ memCopy op)
    struct timespec startS24, endS24;
    clock_gettime(CLOCK_REALTIME,&startS24);

    if(rightTableHash == NULL) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_psum1,gpu_psum,sizeof(int)*hsize,cudaMemcpyDeviceToDevice));

        build_hash_table<<<grid,block>>>(gpu_dim,jNode->rightTable->tupleNum,gpu_psum1,gpu_bucket,hsize);

        if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_dim));

        if(dev_mp == NULL)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum1));
    }
    //Stop for  Step 2.4 - build_hash_table (+ memCopy op)
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS24);
    pp->joinProf_step24_buildhash_kernel_memcopy += (endS24.tv_sec - startS24.tv_sec)* BILLION + endS24.tv_nsec - startS24.tv_nsec;
            
    //Stop for Step 2 - Build hashTable
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS2);
    pp->joinProf_step2_buildHash += (endS2.tv_sec - startS2.tv_sec)* BILLION + endS2.tv_nsec - startS2.tv_nsec;

    //Update stats from the largest join
    if (jNode->leftTable->tupleNum >= pp->join_leftTableSize 
           && jNode->rightTable->tupleNum >= pp->join_rightTableSize){
        pp->join_leftTableSize = jNode->leftTable->tupleNum;
        pp->join_rightTableSize = jNode->rightTable->tupleNum;
    }

    //Start timer for Step 3 - Join
    struct timespec startS3, endS3;
    clock_gettime(CLOCK_REALTIME,&startS3);

   /*
    *  join on GPU
    */

    //Start timer for Step 3.1 - Allocate memory
    struct timespec startS31, endS31;
    clock_gettime(CLOCK_REALTIME,&startS31);

    threadNum = grid.x * block.x;

    if(dev_mp == NULL) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_count,sizeof(int)*threadNum));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_resPsum,sizeof(int)*threadNum));
    } else {
        gpu_count   = (int *) dev_mp->alloc(sizeof(int) * threadNum);
        gpu_resPsum = (int *) dev_mp->alloc(sizeof(int) * threadNum);
    }
    int *gpuFactFilter = NULL, *filterNum = NULL, * JRes = NULL, * filterPsum = NULL;

    dataPos = jNode->leftTable->dataPos[jNode->leftKeyIndex];
    int format = jNode->leftTable->dataFormat[jNode->leftKeyIndex];

    long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
    long filterSize = jNode->leftTable->attrSize[jNode->leftKeyIndex] * jNode->leftTable->tupleNum;

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_fact, foreignKeySize));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact,jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,cudaMemcpyHostToDevice));

    }else if (dataPos == GPU || dataPos == UVA){
        gpu_fact = jNode->leftTable->content[jNode->leftKeyIndex];
    }

    if(dev_mp == NULL) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFactFilter,filterSize));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&filterNum,filterSize));
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&filterPsum,filterSize));
    } else {
        gpuFactFilter = (int *) dev_mp->alloc(filterSize);
        filterNum     = (int *) dev_mp->alloc(filterSize);
        filterPsum    = (int *) dev_mp->alloc(filterSize);
    }
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuFactFilter,0,filterSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(filterNum,0,filterSize));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(filterPsum,0,filterSize));

    //Stop for Step 3.1 - Allocate memory
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS31);
    pp->joinProf_step31_allocateMem += (endS31.tv_sec - startS31.tv_sec)* BILLION + endS31.tv_nsec - startS31.tv_nsec;
    
    //Start timer for Step 3.2 - Exclusive scan
    struct timespec startS32, endS32;
    clock_gettime(CLOCK_REALTIME,&startS32);

    if(format == UNCOMPRESSED)
    {
        count_join_result<<<grid,block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum, gpu_count,filterNum,hsize);
        // thrust::exclusive_scan(thrust::device, filterNum, filterNum + jNode->leftTable->tupleNum, filterPsum);
        scanImpl(filterNum, jNode->leftTable->tupleNum, filterPsum, pp);

    }
    else if(format == DICT){
        int dNum;
        struct dictHeader * dheader = NULL;
        struct dictHeader * gpuDictHeader = NULL;

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuDictHeader, sizeof(struct dictHeader)));

        if(dataPos == MEM || dataPos == MMAP || dataPos == UVA || dataPos == PINNED){
            dheader = (struct dictHeader *) jNode->leftTable->content[jNode->leftKeyIndex];
            dNum = dheader->dictNum;
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));

        }else if (dataPos == GPU){
            dheader = (struct dictHeader *) malloc(sizeof(struct dictHeader));
            memset(dheader,0,sizeof(struct dictHeader));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(dheader,gpu_fact,sizeof(struct dictHeader), cudaMemcpyDeviceToHost));
            dNum = dheader->dictNum;
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));
            free(dheader);
        }

        int * gpuDictFilter;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

        count_join_result_dict<<<grid,block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpuDictHeader, dNum, gpuDictFilter,hsize);

        transform_dict_filter<<<grid,block>>>(gpuDictFilter, gpu_fact, jNode->leftTable->tupleNum, dNum, gpuFactFilter);

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

        filter_count<<<grid,block>>>(jNode->leftTable->tupleNum, gpu_count, gpuFactFilter);

    }else if (format == RLE){

        count_join_result_rle<<<512,64>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum, gpuFactFilter,hsize);

        filter_count<<<grid, block>>>(jNode->leftTable->tupleNum, gpu_count, gpuFactFilter);
    }

    int tmp1, tmp2;

    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp1,&gpu_count[threadNum-1],sizeof(int),cudaMemcpyDeviceToHost));
    scanImpl(gpu_count,threadNum,gpu_resPsum, pp);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp2,&gpu_resPsum[threadNum-1],sizeof(int),cudaMemcpyDeviceToHost));

    res->tupleNum = tmp1 + tmp2;
    //printf("[INFO]Number of join results: %d\n",res->tupleNum);

    //Stop for Step 3.2 - Exclusive scan
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS32);
    pp->joinProf_step32_exclusiveScan += (endS32.tv_sec - startS32.tv_sec)* BILLION + endS32.tv_nsec - startS32.tv_nsec;
    
    //Start timer for Step 3.3 - Prob and memcpy ops
    struct timespec startS33, endS33;
    clock_gettime(CLOCK_REALTIME,&startS33);

    if(dev_mp == NULL)
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&JRes,res->tupleNum * sizeof(int)));
    else
        JRes = (int *) dev_mp->alloc(res->tupleNum * sizeof(int));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(JRes,0,res->tupleNum * sizeof(int)));

    probe_join_result<<<grid,block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum,filterPsum,JRes,hsize);

    //Stop for Step 3.3 - Prob and memcpy ops
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS33);
    pp->joinProf_step33_prob += (endS33.tv_sec - startS33.tv_sec)* BILLION + endS33.tv_nsec - startS33.tv_nsec;
    
    //Stop for Step 3 - Join
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS3);
    pp->joinProf_step3_join += (endS3.tv_sec - startS3.tv_sec)* BILLION + endS3.tv_nsec - startS3.tv_nsec;
    
    //Start timer for Step 4 - De-allocate memory
    struct timespec startS4, endS4;
    clock_gettime(CLOCK_REALTIME,&startS4);

    if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));
    }

    if(dev_mp == NULL && rightTableHash == NULL)
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_bucket));

    for(int i=0; i<res->totalAttr; i++){
        int index, pos;
        long colSize = 0, resSize = 0;
        int leftRight = 0;

        int attrSize, attrType;
        char * table = NULL;
        int found = 0 , dataPos, format;

        if (jNode->keepInGpu[i] == 1)
            res->dataPos[i] = GPU;
        else
            res->dataPos[i] = MEM;

        for(int k=0;k<jNode->leftOutputAttrNum;k++){
            if (jNode->leftPos[k] == i){
                found = 1;
                leftRight = 0;
                pos = k;
                break;
            }
        }
        if(!found){
            for(int k=0;k<jNode->rightOutputAttrNum;k++){
                if(jNode->rightPos[k] == i){
                    found = 1;
                    leftRight = 1;
                    pos = k;
                    break;
                }
            }
        }

        if(leftRight == 0){
            index = jNode->leftOutputIndex[pos];
            dataPos = jNode->leftTable->dataPos[index];
            format = jNode->leftTable->dataFormat[index];

            table = jNode->leftTable->content[index];
            attrSize  = jNode->leftTable->attrSize[index];
            attrType  = jNode->leftTable->attrType[index];
            colSize = jNode->leftTable->attrTotalSize[index];

            resSize = res->tupleNum * attrSize;
        }else{
            index = jNode->rightOutputIndex[pos];
            dataPos = jNode->rightTable->dataPos[index];
            format = jNode->rightTable->dataFormat[index];

            table = jNode->rightTable->content[index];
            attrSize = jNode->rightTable->attrSize[index];
            attrType = jNode->rightTable->attrType[index];
            colSize = jNode->rightTable->attrTotalSize[index];

            resSize = attrSize * res->tupleNum;
            leftRight = 1;
        }

        if(res_mp == NULL)
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_result,resSize));
        else
            gpu_result = (char *) res_mp->alloc(resSize);

        if(leftRight == 0){
            if(format == UNCOMPRESSED){

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }


                joinFact<<<grid,block>>>(filterPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,filterNum,gpu_result);

                // if(attrSize == sizeof(int))
                //     joinFact_int<<<grid,block>>>(filterPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,filterNum,gpu_result);
                // else
                //     joinFact_other<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

            }else if (format == DICT){
                struct dictHeader * dheader = NULL;
                int byteNum;
                struct dictHeader * gpuDictHeader = NULL;

                dheader = (struct dictHeader *)table;
                byteNum = dheader->bitNum/8;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuDictHeader,sizeof(struct dictHeader)));
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                if (attrSize == sizeof(int))
                    joinFact_dict_int<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);
                else
                    joinFact_dict_other<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

            }else if (format == RLE){

                struct rleHeader* rheader = NULL;

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                rheader = (struct rleHeader*)table;

                int dNum = rheader->dictNum;

                char * gpuRle = NULL;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRle, jNode->leftTable->tupleNum * sizeof(int)));

                unpack_rle<<<grid,block>>>(gpu_fact, gpuRle,jNode->leftTable->tupleNum, dNum);

                //joinFact_int<<<grid,block>>>(gpu_resPsum,gpuRle, attrSize, jNode->leftTable->tupleNum,gpuFactFilter,gpu_result);
                joinFact<<<grid, block>>>(gpu_resPsum, gpuRle, attrSize, jNode->leftTable->tupleNum, gpuFactFilter, gpu_result);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));

            }

        }else{
            if(format == UNCOMPRESSED){

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                joinDim<<<grid, block>>>(filterPsum, gpu_fact, attrSize, jNode->leftTable->tupleNum, filterNum, JRes, gpu_result);
                // if(attrType == sizeof(int))
                //     joinDim_int_new<<<grid,block>>>(filterPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, filterNum, JRes, gpu_result);
                // else
                //     joinDim_other<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);

            }else if (format == DICT){
                struct dictHeader * dheader = NULL;
                int byteNum;
                struct dictHeader * gpuDictHeader = NULL;

                dheader = (struct dictHeader *)table;
                byteNum = dheader->bitNum/8;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuDictHeader,sizeof(struct dictHeader)));
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader),cudaMemcpyHostToDevice));
                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                if(attrSize == sizeof(int))
                    joinDim_dict_int<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader,byteNum,attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
                else
                    joinDim_dict_other<<<grid,block>>>(gpu_resPsum,gpu_fact, gpuDictHeader, byteNum, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

            }else if (format == RLE){

                if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED){
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize,cudaMemcpyHostToDevice));
                }else{
                    gpu_fact = table;
                }

                joinDim_rle<<<grid,block>>>(gpu_resPsum,gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,gpu_result);
            }
        }
        
        res->attrTotalSize[i] = resSize;
        res->dataFormat[i] = UNCOMPRESSED;
        if(res->dataPos[i] == MEM){
            res->content[i] = (char *) malloc(resSize);
            memset(res->content[i],0,resSize);
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i],gpu_result,resSize,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_result));

        }else if(res->dataPos[i] == GPU){
            res->content[i] = gpu_result;
        }
        if(dataPos == MEM || dataPos == MMAP || dataPos == PINNED)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));

    }

    if(dev_mp == NULL) {
        CUDA_SAFE_CALL(cudaFree(gpuFactFilter));
        CUDA_SAFE_CALL(cudaFree(filterNum));
        CUDA_SAFE_CALL(cudaFree(filterPsum));

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_count));
        if(rightTableHash == NULL) {
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_hashNum));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum));
        }
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_resPsum));

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(JRes));
    }

    //Stop for Step 4 - De-allocate memory
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS4);
    pp->joinProf_step4_deallocate += (endS4.tv_sec - startS4.tv_sec)* BILLION + endS4.tv_nsec - startS4.tv_nsec;
    
    //Stop total timer
    clock_gettime(CLOCK_REALTIME, &endS0);
    pp->join_totalTime += (endS0.tv_sec - startS0.tv_sec)* BILLION + endS0.tv_nsec - startS0.tv_nsec;
    
    //Increase count call
    pp->join_callTimes++;

    return res;

}
