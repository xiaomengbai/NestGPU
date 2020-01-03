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
#include <cuda.h>
#include <string.h>
#include <time.h>
#include "../include/common.h"
#include "../include/gpuCudaLib.h"
#include "../include/cudaHash.h"
#include "../include/mempool.h"
#include "scanImpl.cu"

/*
 * Combine the group by columns to build the group by keys. 
 */

__global__ static void build_groupby_key(char ** content, int gbColNum, int * gbIndex, int * gbType, int * gbSize, long tupleNum, int * key, int *num, int* groupNum){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(long i = offset; i< tupleNum; i+= stride){
        char buf[128] = {0};
        for (int j=0; j< gbColNum; j++){
            char tbuf[32]={0};
            int index = gbIndex[j];

            if (index == -1){
                gpuItoa(1,tbuf,10);
                gpuStrncat(buf,tbuf,1);

            }else if (gbType[j] == STRING){
                gpuStrncat(buf, content[index] + i*gbSize[j], gbSize[j]);

            }else if (gbType[j] == INT){
                int key = ((int *)(content[index]))[i];
                gpuItoa(key,tbuf,10);
                gpuStrcat(buf,tbuf);
            }
        }
        int hkey = StringHash(buf) % HSIZE;
        key[i]= hkey;
        num[hkey] = 1;
        atomicAdd(&(groupNum[hkey]), 1);
    }
}


/*
 * Count the number of groups 
 */

__global__ static void count_group_num(int *num, int tupleNum, int *totalCount){
        int stride = blockDim.x * gridDim.x;
        int offset = blockIdx.x * blockDim.x + threadIdx.x;
        int localCount = 0;

        for(int i=offset; i<tupleNum; i+= stride){
                if(num[i] == 1){
                        localCount ++;
                }
        }

        atomicAdd(totalCount,localCount);
}

/*
 * Calculate the groupBy expression.
 */

__device__ static float calMathExp(char **content, struct mathExp exp, int pos){
    float res ;

    if(exp.op == NOOP){
        if (exp.opType == CONS)
            res = exp.opValue;
        else if(exp.opType == COLUMN_INTEGER){
            int index = exp.opValue;
            res = ((int *)(content[index]))[pos];
        }else if(exp.opType == COLUMN_DECIMAL){
            int index = exp.opValue;
            res = ((float *)(content[index]))[pos];
        }
    
    }else if(exp.op == PLUS ){
        res = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos) + calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);

    }else if (exp.op == MINUS){
        res = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos) - calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);

    }else if (exp.op == MULTIPLY){
        res = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos) * calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);

    }else if (exp.op == DIVIDE){
        res = calMathExp(content, ((struct mathExp*)exp.exp)[0],pos) / calMathExp(content, ((struct mathExp*)exp.exp)[1], pos);
    }

    return res;
}


__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}

/*
 * group by constant. Currently only support SUM function.
 */

__global__ static void agg_cal_cons(char ** content, int colNum, struct groupByExp* exp, long tupleNum, char ** result){

    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float buf[32];
    for(int i = 0; i < colNum; i++){
        int func = exp[i].func;
        if(func == MAX)
            buf[i] = FLOAT_MIN;
        else if(func == MIN)
            buf[i] = FLOAT_MAX;
        else
            buf[i] = 0;
    }

    for(int i=index;i<tupleNum;i+=stride){
        for(int j=0;j<colNum;j++){
            int func = exp[j].func;
            if (func == SUM){
                float tmpRes = calMathExp(content, exp[j].exp, i);
                buf[j] += tmpRes;
            }else if (func == AVG){

                float tmpRes = calMathExp(content, exp[j].exp, i)/tupleNum;
                buf[j] += tmpRes;
            }else if (func == MAX){

                float tmpRes = calMathExp(content, exp[j].exp, i);
                buf[j] = buf[j] > tmpRes ? buf[j] : tmpRes;
            }else if (func == MIN){

                float tmpRes = calMathExp(content, exp[j].exp, i);
                buf[j] = buf[j] < tmpRes ? buf[j] : tmpRes;
            }
        }
    }

    for(int i=0;i<colNum;i++)
    {
        int func = exp[i].func;
        if (func == SUM)
            atomicAdd(&((float *)result[i])[0], buf[i]);
        else if (func == MAX)
            atomicMaxFloat(&((float *)result[i])[0], buf[i]);
        else if (func == MIN)
            atomicMinFloat(&((float *)result[i])[0], buf[i]);
    }
}

/*
 * gropu by
 */

__global__ static void agg_cal(char ** content, int colNum, struct groupByExp* exp, int * gbType, int * gbSize, long tupleNum, int * key, int *psum, int * groupNum, char ** result){

    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=index;i<tupleNum;i+=stride){

        int hKey = key[i];
        int offset = psum[hKey];

        for(int j=0;j<colNum;j++){
            int func = exp[j].func;
            if(func ==NOOP){
                int type = exp[j].exp.opType;

                if(type == CONS){
                    int value = exp[j].exp.opValue;
                    ((int *)result[j])[offset] = value;
                }else{
                    int index = exp[j].exp.opValue;
                    int attrSize = gbSize[j];
                    if(attrSize == sizeof(int))
                        ((int *)result[j])[offset] = ((int*)content[index])[i];
                    else
                        memcpy(result[j] + offset*attrSize, content[index] + i * attrSize, attrSize);
                }

            }else if (func == SUM ){
                float tmpRes = calMathExp(content, exp[j].exp, i);
                atomicAdd(& ((float *)result[j])[offset], tmpRes);
            }else if (func == MAX ){
                float tmpRes = calMathExp(content, exp[j].exp, i);
                atomicMaxFloat(& ((float *)result[j])[offset], tmpRes);

            }else if (func == MIN ){
                float tmpRes = calMathExp(content, exp[j].exp, i);
                atomicMinFloat(& ((float *)result[j])[offset], tmpRes);

            }else if (func == AVG){
                float tmpRes = calMathExp(content, exp[j].exp, i)/groupNum[hKey];
                atomicAdd(& ((float *)result[j])[offset], tmpRes);
            }
        }
    }
}

__global__ static void init_int_array(int *array, int array_size, int init_value)
{
    int stride = blockDim.x * gridDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = index; i < array_size; i += stride)
        array[i] = init_value;
}
/* 
 * groupBy: group by the data and calculate. 
 * 
 * Prerequisite:
 *  input data are not compressed
 *
 * Input:
 *  gb: the groupby node which contains the input data and groupby information
 *  pp: records the statistics such as kernel execution time 
 *
 * Return:
 *  a new table node
 */

struct tableNode * groupBy(struct groupByNode * gb, struct statistic * pp, const bool use_mempool = false, const bool use_gpu_mempool = false){

    struct timespec start,end;
    clock_gettime(CLOCK_REALTIME,&start);
    int *gpuGbIndex = NULL, gpuTupleNum, gpuGbColNum;
    int *gpuGbType = NULL, *gpuGbSize = NULL;

    int *gpuGbKey = NULL;
    char ** gpuContent = NULL, **column = NULL;

    /*
     * @gbCount: the number of groups
     * gbConstant: whether group by constant
     */

    int gbCount;
    int gbConstant = 0;

    struct tableNode *res = NULL;
    if(!use_mempool) {
        res = (struct tableNode *) malloc(sizeof(struct tableNode));
        CHECK_POINTER(res);
    } else {
        res = (struct tableNode *) alloc_mempool(sizeof(struct tableNode));
        MEMPOOL_CHECK();
    }

    res->tupleSize = gb->tupleSize;
    res->totalAttr = gb->outputAttrNum;

    if(!use_mempool) {
        res->attrType = (int *) malloc(sizeof(int) * res->totalAttr);
        CHECK_POINTER(res->attrType);
        res->attrSize = (int *) malloc(sizeof(int) * res->totalAttr);
        CHECK_POINTER(res->attrSize);
        res->attrTotalSize = (int *) malloc(sizeof(int) * res->totalAttr);
        CHECK_POINTER(res->attrTotalSize);
        res->dataPos = (int *) malloc(sizeof(int) * res->totalAttr);
        CHECK_POINTER(res->dataPos);
        res->dataFormat = (int *) malloc(sizeof(int) * res->totalAttr);
        CHECK_POINTER(res->dataFormat);
        res->content = (char **) malloc(sizeof(char **) * res->totalAttr);
        CHECK_POINTER(res->content);
    } else {
        res->attrType = (int *) alloc_mempool(sizeof(int) * res->totalAttr);
        res->attrSize = (int *) alloc_mempool(sizeof(int) * res->totalAttr);
        res->attrTotalSize = (int *) alloc_mempool(sizeof(int) * res->totalAttr);
        res->dataPos = (int *) alloc_mempool(sizeof(int) * res->totalAttr);
        res->dataFormat = (int *) alloc_mempool(sizeof(int) * res->totalAttr);
        res->content = (char **) alloc_mempool(sizeof(char **) * res->totalAttr);
        MEMPOOL_CHECK();
    }

    for(int i=0;i<res->totalAttr;i++){
        res->attrType[i] = gb->attrType[i];
        res->attrSize[i] = gb->attrSize[i];
        res->dataFormat[i] = UNCOMPRESSED;
    }
    
    gpuTupleNum = gb->table->tupleNum;
    gpuGbColNum = gb->groupByColNum;

    if(gpuGbColNum == 1 && gb->groupByIndex[0] == -1){
        gbConstant = 1;
    }


    dim3 grid(1024);
    dim3 block(128);
    int blockNum = gb->table->tupleNum / block.x + 1;
    if(blockNum < 1024)
        grid = blockNum;

    int *gpu_hashNum = NULL, *gpu_psum = NULL, *gpuGbCount = NULL, *gpu_groupNum = NULL;

    if(!use_gpu_mempool) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuContent, gb->table->totalAttr * sizeof(char *)));
    } else {
        alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuContent, gb->table->totalAttr * sizeof(char *));
        GPU_MEMPOOL_CHECK(gpu_inner_mp);
    }

    if(!use_mempool) {
        column = (char **) malloc(sizeof(char *) * gb->table->totalAttr);
        CHECK_POINTER(column);
    } else {
        column = (char **) alloc_mempool(sizeof(char *) * gb->table->totalAttr);
        MEMPOOL_CHECK();
    }

    for(int i=0;i<gb->table->totalAttr;i++){
        int attrSize = gb->table->attrSize[i];
        if(gb->table->dataPos[i]==MEM){
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)& column[i], attrSize * gb->table->tupleNum));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[i], gb->table->content[i], attrSize *gb->table->tupleNum, cudaMemcpyHostToDevice));

            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &column[i], sizeof(char *), cudaMemcpyHostToDevice));
        }else{
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuContent[i], &gb->table->content[i], sizeof(char *), cudaMemcpyHostToDevice));
        }
    }

    if(gbConstant != 1){

        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbType, sizeof(int) * gb->groupByColNum));
        } else {
            alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuGbType, sizeof(int) * gb->groupByColNum);
            GPU_MEMPOOL_CHECK(gpu_inner_mp);
        }
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbType,gb->groupByType, sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));
        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbSize, sizeof(int) * gb->groupByColNum));
        } else {
            alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuGbSize, sizeof(int) * gb->groupByColNum);
            GPU_MEMPOOL_CHECK(gpu_inner_mp);
        }
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbSize,gb->groupBySize, sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));


        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbKey, gb->table->tupleNum * sizeof(int)));
        } else {
            alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuGbKey, sizeof(int) * gb->groupByColNum);
            GPU_MEMPOOL_CHECK(gpu_inner_mp);
        }

        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbIndex, sizeof(int) * gb->groupByColNum));
        } else {
            alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuGbIndex, sizeof(int) * gb->groupByColNum);
            GPU_MEMPOOL_CHECK(gpu_inner_mp);
        }
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbIndex, gb->groupByIndex,sizeof(int) * gb->groupByColNum, cudaMemcpyHostToDevice));

        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_hashNum,sizeof(int)*HSIZE));
        } else {
            alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpu_hashNum, sizeof(int) * HSIZE);
            GPU_MEMPOOL_CHECK(gpu_inner_mp);
        }
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_hashNum,0,sizeof(int)*HSIZE));

        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpu_groupNum,sizeof(int)*HSIZE));
        } else {
            alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpu_groupNum, sizeof(int) * HSIZE);
            GPU_MEMPOOL_CHECK(gpu_inner_mp);
        }
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_groupNum,0,sizeof(int)*HSIZE));

        build_groupby_key<<<grid,block>>>(gpuContent,gpuGbColNum, gpuGbIndex, gpuGbType,gpuGbSize,gpuTupleNum, gpuGbKey, gpu_hashNum, gpu_groupNum);
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbType));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbSize));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbIndex));
        }

        gbCount = 1;

        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbCount,sizeof(int)));
        } else {
            alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuGbCount, sizeof(int));
            GPU_MEMPOOL_CHECK(gpu_inner_mp);
        }
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuGbCount, 0, sizeof(int)));

        count_group_num<<<grid,block>>>(gpu_hashNum, HSIZE, gpuGbCount);
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gbCount, gpuGbCount, sizeof(int), cudaMemcpyDeviceToHost));

        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_psum,HSIZE*sizeof(int)));
        } else {
            alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpu_psum, HSIZE * sizeof(int));
            GPU_MEMPOOL_CHECK(gpu_inner_mp);
        }
        scanImpl(gpu_hashNum,HSIZE,gpu_psum,pp);

        if (!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbCount));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_hashNum));
        }
    }

    if(gbConstant == 1)
        res->tupleNum = 1;
    else
        res->tupleNum = gbCount;

    //printf("[INFO]Number of groupBy results: %ld\n",res->tupleNum);

    char ** gpuResult = NULL;
    char ** result = NULL;

    if(!use_mempool) {
        result = (char **)malloc(sizeof(char*)*res->totalAttr);
        CHECK_POINTER(result);
    } else {
        result = (char **) alloc_mempool(sizeof(char *) * res->totalAttr);
        MEMPOOL_CHECK();
    }
    if(!use_gpu_mempool) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuResult, sizeof(char *)* res->totalAttr));
    } else {
        alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuResult, sizeof(char *) * res->totalAttr);
        GPU_MEMPOOL_CHECK(gpu_inner_mp);
    }

    for(int i=0; i<res->totalAttr;i++){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&result[i], res->tupleNum * res->attrSize[i]));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemset(result[i], 0, res->tupleNum * res->attrSize[i]));
        res->content[i] = result[i]; 
        res->dataPos[i] = GPU;
        res->attrTotalSize[i] = res->tupleNum * res->attrSize[i];
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&gpuResult[i], &result[i], sizeof(char *), cudaMemcpyHostToDevice));

        if(gb->gbExp[i].func == MIN && res->attrSize[i] == sizeof(int))
            init_int_array<<<grid, block>>>((int *)result[i], res->tupleNum, FLOAT_MAX);
        else if(gb->gbExp[i].func == MAX && res->attrSize[i] == sizeof(int))
            init_int_array<<<grid, block>>>((int *)result[i], res->tupleNum, FLOAT_MIN);
    }


    if(!use_gpu_mempool) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbType, sizeof(int)*res->totalAttr));
    } else {
        alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuGbType, sizeof(int) * res->totalAttr);
        GPU_MEMPOOL_CHECK(gpu_inner_mp);
    }
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbType, res->attrType, sizeof(int)*res->totalAttr, cudaMemcpyHostToDevice));
    if(!use_gpu_mempool) {
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuGbSize, sizeof(int)*res->totalAttr));
    } else {
        alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuGbSize, sizeof(int) * res->totalAttr);
        GPU_MEMPOOL_CHECK(gpu_inner_mp);
    }
    struct groupByExp *gpuGbExp;

    if(!use_gpu_mempool){
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&gpuGbExp, sizeof(struct groupByExp)*res->totalAttr));
    } else {
        alloc_gpu_mempool(&gpu_inner_mp, (char **)&gpuGbExp, sizeof(struct groupByExp) * res->totalAttr);
        GPU_MEMPOOL_CHECK(gpu_inner_mp);
    }
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuGbExp, gb->gbExp, sizeof(struct groupByExp)*res->totalAttr, cudaMemcpyHostToDevice));
    for(int i=0;i<res->totalAttr;i++){
        struct mathExp * tmpMath;
        if(gb->gbExp[i].exp.opNum == 2){
            if(!use_gpu_mempool) {
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&tmpMath, 2* sizeof(struct mathExp)));
            } else {
                alloc_gpu_mempool(&gpu_inner_mp, (char **)&tmpMath, 2 * sizeof(struct mathExp));
                GPU_MEMPOOL_CHECK(gpu_inner_mp);
            }
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(tmpMath,(struct mathExp*)gb->gbExp[i].exp.exp,2*sizeof(struct mathExp), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&(gpuGbExp[i].exp.exp), &tmpMath, sizeof(struct mathExp *), cudaMemcpyHostToDevice));
        }
    }

    gpuGbColNum = res->totalAttr;

    if(gbConstant !=1){
        agg_cal<<<grid,block>>>(gpuContent, gpuGbColNum, gpuGbExp, gpuGbType, gpuGbSize, gpuTupleNum, gpuGbKey, gpu_psum, gpu_groupNum,gpuResult);
        if(!use_gpu_mempool) {
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbKey));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum));
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_groupNum));
        }
    }else
        agg_cal_cons<<<grid,block>>>(gpuContent, gpuGbColNum, gpuGbExp, gpuTupleNum,gpuResult);

    for(int i=0; i<gb->table->totalAttr;i++){
        if(gb->table->dataPos[i]==MEM)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[i]));
    }
    if(!use_mempool)
        free(column);
    if(!use_gpu_mempool) {
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuContent));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbType));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbSize));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuGbExp));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuResult));
    }

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    //printf("GroupBy Time: %lf\n", timeE/(1000*1000));

    return res;
}
