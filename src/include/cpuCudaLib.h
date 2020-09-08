
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

#ifndef __CPU_LIB_H__
#define __CPU_LIB_H__

#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mman.h>
#include "common.h"

static void initTable(struct tableNode * tn){
    assert(tn != NULL);
    tn->totalAttr = 0;
    tn->tupleNum = 0;
    tn->tupleSize = 0;
    tn->attrType = NULL;
    tn->attrSize = NULL;
    tn->attrTotalSize = NULL;
    tn->dataFormat = NULL;
    tn->dataPos = NULL;
    tn->content = NULL;
    tn->colIdxNum = 0;
}

/*
 * Merge the src table into the dst table. The dst table must be initialized.
 * Only consider the case when the data are all in the memory.
 */

static void mergeIntoTable(struct tableNode *dst, struct tableNode * src, struct statistic *pp){

    struct timespec start, end;

    clock_gettime(CLOCK_REALTIME, &start);

    assert(dst != NULL);
    assert(src != NULL);
    dst->totalAttr = src->totalAttr;
    dst->tupleSize = src->tupleSize;
    dst->tupleNum += src->tupleNum;

    if (dst->attrType == NULL){
        dst->attrType = (int *) malloc(sizeof(int) * dst->totalAttr);
        dst->attrSize = (int *) malloc(sizeof(int) * dst->totalAttr);
        dst->attrIndex = (int *) malloc(sizeof(int) * dst->totalAttr);
        dst->attrTotalSize = (int *) malloc(sizeof(int) * dst->totalAttr);
        dst->dataPos = (int *) malloc(sizeof(int) * dst->totalAttr);
        dst->dataFormat = (int *) malloc(sizeof(int) * dst->totalAttr);

        for(int i=0;i<dst->totalAttr;i++){
            dst->attrType[i] = src->attrType[i];
            dst->attrSize[i] = src->attrSize[i];
            dst->attrIndex[i] = src->attrIndex[i];
            dst->attrTotalSize[i] = src->attrTotalSize[i];
            dst->dataPos[i] = MEM;
            dst->dataFormat[i] = src->dataFormat[i];
        }
    }

    if(dst->content == NULL){
        dst->content = (char **) malloc(sizeof(char *) * dst->totalAttr);
        for(int i=0; i<dst->totalAttr; i++){
            int size = dst->attrTotalSize[i];
            if(src->dataPos[i] == PINNED){
                CUDA_SAFE_CALL_NO_SYNC( cudaMallocHost((void **)&(dst->content[i]), size) );
                memset(dst->content[i], 0, size);
                memcpy(dst->content[i], src->content[i], size);
                dst->dataPos[i] = PINNED;
            }else{
                dst->content[i] = (char *) malloc(size);
                memset(dst->content[i], 0 ,size);
                if(src->dataPos[i] == MEM)
                    memcpy(dst->content[i],src->content[i],size);
                else if (src->dataPos[i] == GPU)
                    cudaMemcpy(dst->content[i], src->content[i],size, cudaMemcpyDeviceToHost);
            }
        }
    }else{
        for(int i=0; i<dst->totalAttr; i++){
            dst->attrTotalSize[i] += src->attrTotalSize[i];
            int size = dst->attrTotalSize[i];
            int offset = dst->attrTotalSize[i] - src->attrTotalSize[i];
            int newSize = src->attrTotalSize[i];
            if(dst->dataPos[i] == PINNED){
                char *old = dst->content[i];
                CUDA_SAFE_CALL_NO_SYNC( cudaMallocHost((void **)&(dst->content[i]), size) );
                memset(dst->content[i], 0, size);
                memcpy(dst->content[i], old, offset);
                if(src->dataPos[i] == MEM || src->dataPos[i] == PINNED)
                    memcpy(dst->content[i] + offset, src->content[i], newSize);
                else if (src->dataPos[i] == GPU)
                    CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(dst->content[i] + offset, src->content[i], newSize, cudaMemcpyDeviceToHost) );
                CUDA_SAFE_CALL_NO_SYNC( cudaFreeHost(old) );
            }else{
                dst->content[i] = (char *) realloc(dst->content[i], size);

                if(src->dataPos[i] == MEM)
                    memcpy(dst->content[i] + offset,src->content[i],newSize);
                else if (src->dataPos[i] == GPU)
                    cudaMemcpy(dst->content[i] + offset, src->content[i],newSize, cudaMemcpyDeviceToHost);
            }
        }
    }

    clock_gettime(CLOCK_REALTIME,&end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    pp->total += timeE/(1000*1000) ;
}

static void freeTable(struct tableNode * tn, const bool free_flag = true){
    int i;

    for(i=0;i<tn->totalAttr;i++){
        if(tn->content[i] == NULL)
            continue;
        if(tn->dataPos[i] == MEM)
            free(tn->content[i]);
        else if(tn->dataPos[i] == MMAP)
            munmap(tn->content[i],tn->attrTotalSize[i]);
        else if(tn->dataPos[i] == GPU)
            cudaFree(tn->content[i]);
        else if(tn->dataPos[i] == UVA || tn->dataPos[i] == PINNED)
            cudaFreeHost(tn->content[i]);
    }

    if(free_flag)
        free(tn->attrType);
    tn->attrType = NULL;
    if(free_flag)
        free(tn->attrSize);
    tn->attrSize = NULL;
    if(free_flag)
        free(tn->attrTotalSize);
    tn->attrTotalSize = NULL;
    if(free_flag)
        free(tn->dataFormat);
    tn->dataFormat = NULL;
    if(free_flag)
        free(tn->dataPos);
    tn->dataPos = NULL;
    if(free_flag)
        free(tn->content);
    tn->content = NULL;

    //Free indexed data
    if (tn->colIdxNum > 0){
        free(tn->colIdx);
        tn->colIdx = NULL;
        if(tn->indexPos == MEM){
            for(i=0;i<tn->colIdxNum;i++){
                free(tn->contentIdx[i]);
                free(tn->posIdx[i]);
            }
        }else if(tn->indexPos == GPU){
            for(i=0;i<tn->colIdxNum;i++){
                cudaFree(tn->contentIdx[i]);
                cudaFree(tn->posIdx[i]);
            }
        }else{
            printf("[ERROR] Index cannot be de-allocated. Needs to be on either on host or device!\n");
            exit(-1);
        }
        tn->contentIdx = NULL;
        tn->posIdx = NULL;
        tn->colIdxNum = 0;
    }
}

static void freeScan(struct scanNode * rel, const bool free_flag = true){
    if(free_flag) free(rel->whereIndex);
    rel->whereIndex = NULL;
    if(free_flag) free(rel->outputIndex);
    rel->outputIndex = NULL;
    if(free_flag) free(rel->filter);
    rel->filter = NULL;
    if(free_flag && rel->tn != NULL)
        freeTable(rel->tn);

}

static void freeMathExp(struct mathExp exp, const bool free_flag = true){
    if (exp.exp != 0 && exp.opNum == 2){
        freeMathExp(((struct mathExp *)exp.exp)[0], free_flag);
        freeMathExp(((struct mathExp *)exp.exp)[1], free_flag);
        if(free_flag) free(((struct mathExp *)exp.exp));
        exp.exp = 0;
    }
}

static void freeGroupByNode(struct groupByNode * tn, const bool free_flag = true){
    if(free_flag) free(tn->groupByIndex);
    tn->groupByIndex = NULL;
    for (int i=0;i<tn->outputAttrNum;i++){
        freeMathExp(tn->gbExp[i].exp, free_flag);
    }
    if(free_flag) free(tn->gbExp);
    tn->gbExp = NULL;
    if(free_flag && tn->table != NULL)
        freeTable(tn->table);
}

static void freeOrderByNode(struct orderByNode * tn, const bool free_flag = true){
    if(free_flag) free(tn->orderBySeq);
    tn->orderBySeq = NULL;
    if(free_flag) free(tn->orderByIndex);
    tn->orderByIndex = NULL;
    if(free_flag && tn->table != NULL)
        freeTable(tn->table);
}

static void printCol(char *col, int size, int type,int tupleNum,int pos){
    if (pos ==GPU){
        if(type == INT){
            int * cpuCol = (int *)malloc(size * tupleNum);
            cudaMemcpy(cpuCol,col,size * tupleNum, cudaMemcpyDeviceToHost);
            for(int i=0;i<tupleNum;i++){
                printf("%d\n", ((int*)cpuCol)[i]);
            }
            free(cpuCol);
        }else if (type == FLOAT){
            float * cpuCol = (float *)malloc(size * tupleNum);
            cudaMemcpy(cpuCol,col,size * tupleNum, cudaMemcpyDeviceToHost);
            for(int i=0;i<tupleNum;i++){
                printf("%f\n", ((float*)cpuCol)[i]);
            }
            free(cpuCol);

        }else if (type == STRING){

            char * cpuCol = (char *)malloc(size * tupleNum);
            cudaMemcpy(cpuCol,col,size * tupleNum, cudaMemcpyDeviceToHost);
            for(int i=0;i<tupleNum;i++){
                char tbuf[128] = {0};
                memset(tbuf,0,sizeof(tbuf));
                memcpy(tbuf,cpuCol + i*size, size);
                printf("%s\n", tbuf);
            }
            free(cpuCol);
        }
    }else if (pos == MEM){
        if(type == INT){
            int * cpuCol = (int*)col;
            for(int i=0;i<tupleNum;i++){
                printf("%d\n", ((int*)cpuCol)[i]);
            }

        }else if (type == FLOAT){

            float * cpuCol = (float*)col;
            for(int i=0;i<tupleNum;i++){
                printf("%f\n", ((float*)cpuCol)[i]);
            }
        }else if (type == STRING){
            char * cpuCol = col;
            for(int i=0;i<tupleNum;i++){
                char tbuf[128] = {0};
                memset(tbuf,0,sizeof(tbuf));
                memcpy(tbuf,cpuCol + i*size, size);
                printf("%s\n", tbuf);
            }

        }
    }
}

static void printMaterializedTable(struct materializeNode &mn, char *table)
{
    char buf[20];
    time_t t;
    for(int i = 0; i < mn.table->tupleNum; i++){
        size_t tuple_off = 0;
        printf("Tuple %d: ", i);
        for(int j = 0; j < mn.table->totalAttr; j++){
            switch(mn.table->attrType[j]){
            case DATE:
                t = *(int *)(table + i * mn.table->tupleSize + tuple_off);
                memset(buf, 0, 20);
                strftime(buf, 20, "%Y-%m-%d", gmtime(&t));
                printf("(date) %s, ", buf);
                break;
            case INT:
                printf("(int) %d, ", *(int *)(table + i * mn.table->tupleSize + tuple_off)); break;
            case FLOAT:
                printf("(float) %f, ", *(float *)(table + i * mn.table->tupleSize + tuple_off)); break;
            case STRING:
                size_t strlen = mn.table->attrSize[j];
                char *strout = (char *)malloc(strlen + 1);
                strout[strlen] = '\0';
                memcpy(strout, table + i * mn.table->tupleSize + tuple_off, strlen);
                printf("(string) %s, ", strout);
                free(strout);
                break;
            }
            tuple_off += mn.table->attrSize[j];
        }
        printf("\n");
    }
}

static void transferTableColumnToGPU(struct tableNode *tn, int i)
{
    if(tn->totalAttr > i && (tn->dataPos[i] == MEM || tn->dataPos[i] == PINNED || tn->dataPos[i] == MMAP)) {
        char *tmp;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&tmp, tn->attrTotalSize[i]));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(tmp, tn->content[i], tn->attrTotalSize[i], cudaMemcpyHostToDevice));
        if (tn->dataPos[i] == PINNED)
            CUDA_SAFE_CALL_NO_SYNC( cudaFreeHost(tn->content[i]) );
        else
            free(tn->content[i]);
        tn->content[i] = tmp;
        tn->dataPos[i] = GPU;
    }
}

static void transferTableColumnToCPU(struct tableNode *tn, int i)
{
    if(tn->totalAttr > i && tn->dataPos[i] == GPU) {
        char *tmp = (char *)malloc(tn->attrTotalSize[i]);
        assert(tmp != NULL);
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(tmp, tn->content[i], tn->attrTotalSize[i], cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(tn->content[i]));
        tn->content[i] = tmp;
        tn->dataPos[i] = MEM;
    }
}

static int tableScanGPUMemSize(struct scanNode *sn)
{
    int memsize = 0;
    dim3 grid(2048);
    dim3 block(256);
    int tupleNum = sn->tn->tupleNum;
    int blockNum = sn->tn->tupleNum / block.x + 1;
    if(blockNum<2048)
        grid = blockNum;
    int threadNum = grid.x * block.x;

    memsize += sizeof(int) * tupleNum;
    memsize += 2 * sizeof(int) * threadNum;
    memsize += sizeof(struct whereExp);

    return memsize;
};

static int hashJoinGPUMemSize(joinNode *jn, bool prehash)
{
    int memsize = 0;
    int hsize = jn->rightTable->tupleNum;
    long primaryKeySize = sizeof(int) * jn->rightTable->tupleNum;
    NP2(hsize);
    int defaultBlock = 4096;

    dim3 grid(defaultBlock);
    dim3 block(256);

    int blockNum;
    int threadNum;

    blockNum = jn->leftTable->tupleNum / block.x + 1;
    if(blockNum < defaultBlock)
        grid = blockNum;
    else
        grid = defaultBlock;

    threadNum = grid.x * block.x;

    long filterSize = jn->leftTable->attrSize[jn->leftKeyIndex] * jn->leftTable->tupleNum;

    if(!prehash) {
        memsize += sizeof(int) * hsize; // gpu_hashNum
        memsize += hsize * sizeof(int); // gpu_psum + gpu_psum1
        memsize += 2 * primaryKeySize; // gpu_bucket
    }

    memsize += sizeof(int) * threadNum * 2; // gpu_count + gpu_resPsum
    memsize += filterSize * 3; // gpuFactFilter + filterNum + filterPsum
    memsize += sizeof(int) * (jn->leftTable->tupleNum + jn->rightTable->tupleNum); // JRes

    return memsize;
}

static size_t groupByGPUMemSize(struct groupByNode *gb)
{
    size_t memsize = 0;

    int gpuGbColNum;
    int gbConstant = 0;

    gpuGbColNum = gb->groupByColNum;

    if(gpuGbColNum == 1 && gb->groupByIndex[0] == -1){
        gbConstant = 1;
    }

    memsize += gb->table->totalAttr * sizeof(char *); //gpuContent
    size_t alignedGpuTupleNum = gb->table->tupleNum;
    NP2(alignedGpuTupleNum);

    if(gbConstant != 1){
        memsize += sizeof(int) * (size_t) gb->groupByColNum * 3; // gpuGbType + gpuGbSize + gpuGbIndex
        memsize += sizeof(int) * (size_t) alignedGpuTupleNum; // gpuGbKey
        memsize += sizeof(int) * (size_t) HSIZE * 3; // gpu_hashNum + gpu_groupNum + gpu_psum
        memsize += sizeof(int); // gpuGbCount
    }
    memsize += sizeof(char *) * gb->outputAttrNum; // gpuResult
    memsize += sizeof(int) * gb->outputAttrNum * 2; // gpuGbType + gpuGbSize
    memsize += sizeof(struct groupByExp) * gb->outputAttrNum; // gpuGbExp;
    for(int i = 0;i < gb->outputAttrNum; i++)
        if(gb->gbExp[i].exp.opNum == 2)
            memsize += 2 * sizeof(struct mathExp); // tmpMath

    return memsize;
}
#endif
