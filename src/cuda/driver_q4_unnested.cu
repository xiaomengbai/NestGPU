/* This file is generated by code_gen.py */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <malloc.h>
#include <time.h>
#include <getopt.h>
#include <string.h>
#include <linux/limits.h>
#include "../include/common.h"
#include "../include/hashJoin.h"
#include "../include/schema.h"
#include "../include/Mempool.h"
#include <map>
#include "../include/cpuCudaLib.h"
#include "../include/gpuCudaLib.h"
extern struct tableNode* tableScan(struct scanNode *,struct statistic *, Mempool *, Mempool *, Mempool *, int *, int *, int *);
extern void createIndex (struct tableNode *, int, int, struct statistic *);
extern struct tableNode* hashJoin(struct joinNode *, struct statistic *, Mempool *, Mempool *, Mempool *, int *);
extern int *buildColumnHash(struct tableNode *, int, struct statistic *);
extern struct tableNode* groupBy(struct groupByNode *,struct statistic *, Mempool *, Mempool *);
extern struct tableNode* orderBy(struct orderByNode *, struct statistic *);
extern char* materializeCol(struct materializeNode * mn, struct statistic *);

#define CHECK_POINTER(p) do {\
    if(p == NULL){   \
        perror("Failed to allocate host memory");    \
        exit(-1);      \
    }} while(0)

int main(int argc, char ** argv){

    /* For initializing CUDA device */
    int * cudaTmp;
    cudaMalloc((void**)&cudaTmp,sizeof(int));
    cudaFree(cudaTmp);

    int table;
    int long_index;
    char path[PATH_MAX];
    int setPath = 0;
    struct option long_options[] = {
        {"datadir",required_argument,0,'0'}
    };

    while((table=getopt_long(argc,argv,"",long_options,&long_index))!=-1){
        switch(table){
            case '0':
                setPath = 1;
                strcpy(path,optarg);
                break;
        }
    }

    if(setPath == 1)
        chdir(path);

    struct timespec start, end;
    struct timespec diskStart, diskEnd;
    double diskTotal = 0;
    clock_gettime(CLOCK_REALTIME,&start);
    struct statistic pp;
    pp.total = pp.kernel = pp.pcie = 0;

    pp.outerTableSize = pp.resultTableSize = 0;

    pp.buildIndexTotal = 0;
    pp.tableScanTotal = 0;
    pp.tableScanCount = 0;
    pp.whereMemCopy_s1 = 0;
    pp.dataMemCopy_s2 = 0;
    pp.scanTotal_s3 = 0;
    pp.preScanTotal_s4 = 0;
    pp.preScanCount_s4 = 0;
    pp.preScanResultMemCopy_s5 = 0;
    pp.dataMemCopyOther_s6 = 0;
    pp.materializeResult_s7 = 0;
    pp.finalResultMemCopy_s8 = 0;
    pp.create_tableNode_S01 = 0;
    pp.mallocRes_S02 = 0;
    pp.deallocateBuffs_S03 = 0;
    pp.getIndexPos_idxS1 = 0;
    pp.getRange_idxS2 = 0;
    pp.convertMemToElement_idxS3 = 0;
    pp.getMapping_idxS4 = 0;
    pp.setBitmapZeros_idxS5 = 0;
    pp.buildBitmap_idxS6 = 0;
    pp.countScanKernel_countS1 = 0;
    pp.scanImpl_countS2 = 0;
    pp.preallocBlockSums_scanImpl_S1 = 0;
    pp.prescanArray_scanImpl_S2 = 0;
    pp.deallocBlockSums_scanImpl_S3 = 0;
    pp.setVar_prescan_S1 = 0;
    pp.preScanKernel_prescan_S2 = 0;
    pp.uniformAddKernel_prescan_S3 = 0;

    pp.join_totalTime = 0;
    pp.join_callTimes = 0;
    pp.join_leftTableSize = 0;
    pp.join_rightTableSize = 0;
    pp.joinProf_step1_allocateMem = 0;
    pp.joinProf_step2_buildHash = 0;
    pp.joinProf_step21_allocateMem = 0;
    pp.joinProf_step22_Count_hash_num = 0;
    pp.joinProf_step23_scanImpl = 0;
    pp.joinProf_step24_buildhash_kernel_memcopy = 0;
    pp.joinProf_step3_join = 0;
    pp.joinProf_step31_allocateMem = 0;
    pp.joinProf_step32_exclusiveScan = 0;
    pp.joinProf_step33_prob = 0;
    pp.joinProf_step4_deallocate = 0;

    pp.groupby_totalTime = 0;
    pp.groupby_callTimes = 0;
    pp.groupby_step1_allocMem = 0;
    pp.groupby_step2_copyToDevice = 0;
    pp.groupby_step3_buildGroupByKey = 0;
    pp.groupby_step4_groupCount = 0;
    pp.groupby_step5_AllocRes = 0;
    pp.groupby_step6_copyDataCols = 0;
    pp.groupby_step7_computeAgg = 0;
    pp.groupby_step8_deallocate = 0;

    Mempool host_mempool(MEM);
    Mempool gpu_inner_mp(GPU);
    Mempool gpu_inter_mp(GPU);
    printf("gpu_inner_mp.freesize() is %lu\n", gpu_inner_mp.freesize());

    // Load columns from the table LINEITEM
    struct tableNode *lineitemTable;
    lineitemTable = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
    initTable(lineitemTable);
    {
        struct tableNode *_lineitem_table;
        int outFd;
        long outSize;
        char *outTable;
        long offset, tupleOffset;
        int blockTotal;
        struct columnHeader header;

        // Retrieve the block number from LINEITEM0
        outFd = open("LINEITEM0", O_RDONLY);
        read(outFd, &header, sizeof(struct columnHeader));
        blockTotal = header.blockTotal;
        close(outFd);
        offset = 0;
        tupleOffset = 0;
        for(int i = 0; i < blockTotal; i++){

            // Table initialization
            _lineitem_table = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
            _lineitem_table->totalAttr = 3;
            _lineitem_table->attrType = (int *)host_mempool.alloc(sizeof(int) * 3);
            _lineitem_table->attrSize = (int *)host_mempool.alloc(sizeof(int) * 3);
            _lineitem_table->attrIndex = (int *)host_mempool.alloc(sizeof(int) * 3);
            _lineitem_table->attrTotalSize = (int *)host_mempool.alloc(sizeof(int) * 3);
            _lineitem_table->dataPos = (int *)host_mempool.alloc(sizeof(int) * 3);
            _lineitem_table->dataFormat = (int *)host_mempool.alloc(sizeof(int) * 3);
            _lineitem_table->content = (char **)host_mempool.alloc(sizeof(char *) * 3);

            // Load column 0, type: INTEGER
            _lineitem_table->attrSize[0] = sizeof(int);
            _lineitem_table->attrIndex[0] = 0;
            _lineitem_table->attrType[0] = INT;
            _lineitem_table->dataPos[0] = MEM;
            outFd = open("LINEITEM0", O_RDONLY);
            offset = i * sizeof(struct columnHeader) + tupleOffset * sizeof(int);
            lseek(outFd, offset, SEEK_SET);
            read(outFd, &header, sizeof(struct columnHeader));
            offset += sizeof(struct columnHeader);
            _lineitem_table->dataFormat[0] = header.format;
            outSize = header.tupleNum * sizeof(int);
            _lineitem_table->attrTotalSize[0] = outSize;

            clock_gettime(CLOCK_REALTIME,&diskStart);
            outTable =(char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);
            _lineitem_table->content[0] = (char *)memalign(256, outSize);
            memcpy(_lineitem_table->content[0], outTable, outSize);
            munmap(outTable, outSize);
            clock_gettime(CLOCK_REALTIME, &diskEnd);
            diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;
            close(outFd);

            // Load column 11, type: DATE
            _lineitem_table->attrSize[1] = sizeof(int);
            _lineitem_table->attrIndex[1] = 11;
            _lineitem_table->attrType[1] = DATE;
            _lineitem_table->dataPos[1] = MEM;
            outFd = open("LINEITEM11", O_RDONLY);
            offset = i * sizeof(struct columnHeader) + tupleOffset * sizeof(int);
            lseek(outFd, offset, SEEK_SET);
            read(outFd, &header, sizeof(struct columnHeader));
            offset += sizeof(struct columnHeader);
            _lineitem_table->dataFormat[1] = header.format;
            outSize = header.tupleNum * sizeof(int);
            _lineitem_table->attrTotalSize[1] = outSize;

            clock_gettime(CLOCK_REALTIME,&diskStart);
            outTable =(char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);
            _lineitem_table->content[1] = (char *)memalign(256, outSize);
            memcpy(_lineitem_table->content[1], outTable, outSize);
            munmap(outTable, outSize);
            clock_gettime(CLOCK_REALTIME, &diskEnd);
            diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;
            close(outFd);

            // Load column 12, type: DATE
            _lineitem_table->attrSize[2] = sizeof(int);
            _lineitem_table->attrIndex[2] = 12;
            _lineitem_table->attrType[2] = DATE;
            _lineitem_table->dataPos[2] = MEM;
            outFd = open("LINEITEM12", O_RDONLY);
            offset = i * sizeof(struct columnHeader) + tupleOffset * sizeof(int);
            lseek(outFd, offset, SEEK_SET);
            read(outFd, &header, sizeof(struct columnHeader));
            offset += sizeof(struct columnHeader);
            _lineitem_table->dataFormat[2] = header.format;
            outSize = header.tupleNum * sizeof(int);
            _lineitem_table->attrTotalSize[2] = outSize;

            clock_gettime(CLOCK_REALTIME,&diskStart);
            outTable =(char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);
            _lineitem_table->content[2] = (char *)memalign(256, outSize);
            memcpy(_lineitem_table->content[2], outTable, outSize);
            munmap(outTable, outSize);
            clock_gettime(CLOCK_REALTIME, &diskEnd);
            diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;
            close(outFd);

            _lineitem_table->tupleSize = 0 + sizeof(int) + sizeof(int) + sizeof(int);
            _lineitem_table->tupleNum = header.tupleNum;

            if(blockTotal != 1){
                mergeIntoTable(lineitemTable,_lineitem_table, &pp);
                clock_gettime(CLOCK_REALTIME, &diskStart);
                freeTable(_lineitem_table);
                clock_gettime(CLOCK_REALTIME, &diskEnd);
                diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;
            }else{
                lineitemTable = _lineitem_table;
            }
            tupleOffset += header.tupleNum;
        }
        lineitemTable->colIdxNum = 0;
        _lineitem_table->keepInGpuIdx = 1;
    }

    // Load columns from the table ORDERS
    struct tableNode *ordersTable;
    ordersTable = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
    initTable(ordersTable);
    {
        struct tableNode *_orders_table;
        int outFd;
        long outSize;
        char *outTable;
        long offset, tupleOffset;
        int blockTotal;
        struct columnHeader header;

        // Retrieve the block number from ORDERS5
        outFd = open("ORDERS5", O_RDONLY);
        read(outFd, &header, sizeof(struct columnHeader));
        blockTotal = header.blockTotal;
        close(outFd);
        offset = 0;
        tupleOffset = 0;
        for(int i = 0; i < blockTotal; i++){

            // Table initialization
            _orders_table = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
            _orders_table->totalAttr = 3;
            _orders_table->attrType = (int *)host_mempool.alloc(sizeof(int) * 3);
            _orders_table->attrSize = (int *)host_mempool.alloc(sizeof(int) * 3);
            _orders_table->attrIndex = (int *)host_mempool.alloc(sizeof(int) * 3);
            _orders_table->attrTotalSize = (int *)host_mempool.alloc(sizeof(int) * 3);
            _orders_table->dataPos = (int *)host_mempool.alloc(sizeof(int) * 3);
            _orders_table->dataFormat = (int *)host_mempool.alloc(sizeof(int) * 3);
            _orders_table->content = (char **)host_mempool.alloc(sizeof(char *) * 3);

            // Load column 5, type: TEXT
            _orders_table->attrSize[0] = 15;
            _orders_table->attrIndex[0] = 5;
            _orders_table->attrType[0] = STRING;
            _orders_table->dataPos[0] = MEM;
            outFd = open("ORDERS5", O_RDONLY);
            offset = i * sizeof(struct columnHeader) + tupleOffset * 15;
            lseek(outFd, offset, SEEK_SET);
            read(outFd, &header, sizeof(struct columnHeader));
            offset += sizeof(struct columnHeader);
            _orders_table->dataFormat[0] = header.format;
            outSize = header.tupleNum * 15;
            _orders_table->attrTotalSize[0] = outSize;

            clock_gettime(CLOCK_REALTIME,&diskStart);
            outTable =(char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);
            _orders_table->content[0] = (char *)memalign(256, outSize);
            memcpy(_orders_table->content[0], outTable, outSize);
            munmap(outTable, outSize);
            clock_gettime(CLOCK_REALTIME, &diskEnd);
            diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;
            close(outFd);

            // Load column 0, type: INTEGER
            _orders_table->attrSize[1] = sizeof(int);
            _orders_table->attrIndex[1] = 0;
            _orders_table->attrType[1] = INT;
            _orders_table->dataPos[1] = MEM;
            outFd = open("ORDERS0", O_RDONLY);
            offset = i * sizeof(struct columnHeader) + tupleOffset * sizeof(int);
            lseek(outFd, offset, SEEK_SET);
            read(outFd, &header, sizeof(struct columnHeader));
            offset += sizeof(struct columnHeader);
            _orders_table->dataFormat[1] = header.format;
            outSize = header.tupleNum * sizeof(int);
            _orders_table->attrTotalSize[1] = outSize;

            clock_gettime(CLOCK_REALTIME,&diskStart);
            outTable =(char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);
            _orders_table->content[1] = (char *)memalign(256, outSize);
            memcpy(_orders_table->content[1], outTable, outSize);
            munmap(outTable, outSize);
            clock_gettime(CLOCK_REALTIME, &diskEnd);
            diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;
            close(outFd);

            // Load column 4, type: DATE
            _orders_table->attrSize[2] = sizeof(int);
            _orders_table->attrIndex[2] = 4;
            _orders_table->attrType[2] = DATE;
            _orders_table->dataPos[2] = MEM;
            outFd = open("ORDERS4", O_RDONLY);
            offset = i * sizeof(struct columnHeader) + tupleOffset * sizeof(int);
            lseek(outFd, offset, SEEK_SET);
            read(outFd, &header, sizeof(struct columnHeader));
            offset += sizeof(struct columnHeader);
            _orders_table->dataFormat[2] = header.format;
            outSize = header.tupleNum * sizeof(int);
            _orders_table->attrTotalSize[2] = outSize;

            clock_gettime(CLOCK_REALTIME,&diskStart);
            outTable =(char *)mmap(0, outSize, PROT_READ, MAP_SHARED, outFd, offset);
            _orders_table->content[2] = (char *)memalign(256, outSize);
            memcpy(_orders_table->content[2], outTable, outSize);
            munmap(outTable, outSize);
            clock_gettime(CLOCK_REALTIME, &diskEnd);
            diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;
            close(outFd);

            _orders_table->tupleSize = 0 + 15 + sizeof(int) + sizeof(int);
            _orders_table->tupleNum = header.tupleNum;

            if(blockTotal != 1){
                mergeIntoTable(ordersTable,_orders_table, &pp);
                clock_gettime(CLOCK_REALTIME, &diskStart);
                freeTable(_orders_table);
                clock_gettime(CLOCK_REALTIME, &diskEnd);
                diskTotal += (diskEnd.tv_sec -  diskStart.tv_sec)* BILLION + diskEnd.tv_nsec - diskStart.tv_nsec;
            }else{
                ordersTable = _orders_table;
            }
            tupleOffset += header.tupleNum;
        }
        ordersTable->colIdxNum = 0;
        _orders_table->keepInGpuIdx = 1;
    }



    struct tableNode *result;

    // Process the TableNode for ORDERS
    printf("process the tableNode for orders\n");
    struct tableNode *od0;
    od0 = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
    initTable(od0);
    {
        char * subqRes0;
        char *free_pos;
        char *free_gpu_pos;
        struct tableNode *ordersTablePartial;
        ordersTablePartial = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
        ordersTablePartial->totalAttr = 3;
        ordersTablePartial->attrType = (int *)host_mempool.alloc(sizeof(int) * 3);
        ordersTablePartial->attrSize = (int *)host_mempool.alloc(sizeof(int) * 3);
        ordersTablePartial->attrIndex = (int *)host_mempool.alloc(sizeof(int) * 3);
        ordersTablePartial->attrTotalSize = (int *)host_mempool.alloc(sizeof(int) * 3);
        ordersTablePartial->dataPos = (int *)host_mempool.alloc(sizeof(int) * 3);
        ordersTablePartial->dataFormat = (int *)host_mempool.alloc(sizeof(int) * 3);
        ordersTablePartial->content = (char **)host_mempool.alloc(sizeof(char *) * 3);
        int tuple_size = 0;
        ordersTablePartial->attrSize[0] = ordersTable->attrSize[0];
        ordersTablePartial->attrIndex[0] = ordersTable->attrIndex[0];
        ordersTablePartial->attrType[0] = ordersTable->attrType[0];
        ordersTablePartial->dataPos[0] = ordersTable->dataPos[0];
        ordersTablePartial->dataFormat[0] = ordersTable->dataFormat[0];
        ordersTablePartial->attrTotalSize[0] = ordersTable->attrTotalSize[0];
        ordersTablePartial->content[0] = ordersTable->content[0];
        tuple_size += ordersTablePartial->attrSize[0];

        ordersTablePartial->attrSize[1] = ordersTable->attrSize[1];
        ordersTablePartial->attrIndex[1] = ordersTable->attrIndex[1];
        ordersTablePartial->attrType[1] = ordersTable->attrType[1];
        ordersTablePartial->dataPos[1] = ordersTable->dataPos[1];
        ordersTablePartial->dataFormat[1] = ordersTable->dataFormat[1];
        ordersTablePartial->attrTotalSize[1] = ordersTable->attrTotalSize[1];
        ordersTablePartial->content[1] = ordersTable->content[1];
        tuple_size += ordersTablePartial->attrSize[1];

        ordersTablePartial->attrSize[2] = ordersTable->attrSize[2];
        ordersTablePartial->attrIndex[2] = ordersTable->attrIndex[2];
        ordersTablePartial->attrType[2] = ordersTable->attrType[2];
        ordersTablePartial->dataPos[2] = ordersTable->dataPos[2];
        ordersTablePartial->dataFormat[2] = ordersTable->dataFormat[2];
        ordersTablePartial->attrTotalSize[2] = ordersTable->attrTotalSize[2];
        ordersTablePartial->content[2] = ordersTable->content[2];
        tuple_size += ordersTablePartial->attrSize[2];

        ordersTablePartial->tupleSize = tuple_size;
        ordersTablePartial->tupleNum = ordersTable->tupleNum;

        ordersTablePartial->colIdxNum = 0;
        ordersTablePartial->keepInGpuIdx = 1;
        // Where conditions: AND(GTH(ORDERS.4,"1996-01-15"),LEQ(ORDERS.4,"1996-04-15"))
        struct scanNode ordersRel;
        ordersRel.tn = ordersTablePartial;
        ordersRel.hasWhere = 1;
        ordersRel.whereAttrNum = 1;
        ordersRel.whereIndex = (int *)host_mempool.alloc(sizeof(int) * 2);
        ordersRel.outputIndex = (int *)host_mempool.alloc(sizeof(int) * 2);
        ordersRel.outputNum = 2;
        ordersRel.outputIndex[0] = 0;
        ordersRel.outputIndex[1] = 1;
        ordersRel.whereIndex[0] = 2;
        ordersRel.keepInGpu = 1;
        ordersRel.filter = (struct whereCondition *)host_mempool.alloc(sizeof(struct whereCondition));
        (ordersRel.filter)->nested = 0;
        (ordersRel.filter)->expNum = 2;
        (ordersRel.filter)->exp = (struct whereExp*)host_mempool.alloc(sizeof(struct whereExp) *2);
        (ordersRel.filter)->andOr = AND;
        (ordersRel.filter)->exp[0].index    = 0;
        (ordersRel.filter)->exp[0].relation = GTH;
        (ordersRel.filter)->exp[0].dataPos  = MEM;
        {
            struct tm tm; memset(&tm, 0, sizeof(struct tm));
            strptime("1996-01-15", "%Y-%m-%d", &tm);
            int tmp = mktime(&tm);
            memcpy((ordersRel.filter)->exp[0].content, &tmp, sizeof(int));
        }
        (ordersRel.filter)->exp[1].index    = 0;
        (ordersRel.filter)->exp[1].relation = LEQ;
        (ordersRel.filter)->exp[1].dataPos  = MEM;
        {
            struct tm tm; memset(&tm, 0, sizeof(struct tm));
            strptime("1996-04-15", "%Y-%m-%d", &tm);
            int tmp = mktime(&tm);
            memcpy((ordersRel.filter)->exp[1].content, &tmp, sizeof(int));
        }
        int dev_memsize = tableScanGPUMemSize(&ordersRel);
        if(gpu_inner_mp.freesize() < dev_memsize)
            gpu_inner_mp.resize(gpu_inner_mp.usedsize() + dev_memsize);
        char *origin_pos = gpu_inner_mp.freepos();
        od0 = tableScan(&ordersRel, &pp, &host_mempool, &gpu_inner_mp, NULL, NULL, NULL, NULL);
        gpu_inner_mp.freeto(origin_pos);

        clock_gettime(CLOCK_REALTIME, &diskStart);
        ordersTablePartial->content[0] = NULL;
        ordersTablePartial->content[1] = NULL;
        ordersTablePartial->content[2] = NULL;
        freeScan(&ordersRel, false);

        clock_gettime(CLOCK_REALTIME, &diskEnd);
        od0->colIdxNum = 0;
    }

    // Process the TableNode for LINEITEM
    printf("process the TableNode for LINEITEM\n");
    struct tableNode *li0;
    li0 = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
    initTable(li0);

    {
        char * subqRes0;
        char *free_pos;
        char *free_gpu_pos;
        struct tableNode *lineitemTablePartial;
        lineitemTablePartial = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
        lineitemTablePartial->totalAttr = 3;
        lineitemTablePartial->attrType = (int *)host_mempool.alloc(sizeof(int) * 3);
        lineitemTablePartial->attrSize = (int *)host_mempool.alloc(sizeof(int) * 3);
        lineitemTablePartial->attrIndex = (int *)host_mempool.alloc(sizeof(int) * 3);
        lineitemTablePartial->attrTotalSize = (int *)host_mempool.alloc(sizeof(int) * 3);
        lineitemTablePartial->dataPos = (int *)host_mempool.alloc(sizeof(int) * 3);
        lineitemTablePartial->dataFormat = (int *)host_mempool.alloc(sizeof(int) * 3);
        lineitemTablePartial->content = (char **)host_mempool.alloc(sizeof(char *) * 3);
        int tuple_size = 0;
        lineitemTablePartial->attrSize[0] = lineitemTable->attrSize[0];
        lineitemTablePartial->attrIndex[0] = lineitemTable->attrIndex[0];
        lineitemTablePartial->attrType[0] = lineitemTable->attrType[0];
        lineitemTablePartial->dataPos[0] = lineitemTable->dataPos[0];
        lineitemTablePartial->dataFormat[0] = lineitemTable->dataFormat[0];
        lineitemTablePartial->attrTotalSize[0] = lineitemTable->attrTotalSize[0];
        lineitemTablePartial->content[0] = lineitemTable->content[0];
        tuple_size += lineitemTablePartial->attrSize[0];

        lineitemTablePartial->attrSize[1] = lineitemTable->attrSize[1];
        lineitemTablePartial->attrIndex[1] = lineitemTable->attrIndex[1];
        lineitemTablePartial->attrType[1] = lineitemTable->attrType[1];
        lineitemTablePartial->dataPos[1] = lineitemTable->dataPos[1];
        lineitemTablePartial->dataFormat[1] = lineitemTable->dataFormat[1];
        lineitemTablePartial->attrTotalSize[1] = lineitemTable->attrTotalSize[1];
        lineitemTablePartial->content[1] = lineitemTable->content[1];
        tuple_size += lineitemTablePartial->attrSize[1];

        lineitemTablePartial->attrSize[2] = lineitemTable->attrSize[2];
        lineitemTablePartial->attrIndex[2] = lineitemTable->attrIndex[2];
        lineitemTablePartial->attrType[2] = lineitemTable->attrType[2];
        lineitemTablePartial->dataPos[2] = lineitemTable->dataPos[2];
        lineitemTablePartial->dataFormat[2] = lineitemTable->dataFormat[2];
        lineitemTablePartial->attrTotalSize[2] = lineitemTable->attrTotalSize[2];
        lineitemTablePartial->content[2] = lineitemTable->content[2];
        tuple_size += lineitemTablePartial->attrSize[2];

        lineitemTablePartial->tupleSize = tuple_size;
        lineitemTablePartial->tupleNum = lineitemTable->tupleNum;

        lineitemTablePartial->colIdxNum = 0;
        lineitemTablePartial->keepInGpuIdx = 1;
        // Where conditions: LTH(LINEITEM.11,LINEITEM.12)
        struct scanNode lineitemRel;
        lineitemRel.tn = lineitemTablePartial;
        lineitemRel.hasWhere = 1;
        lineitemRel.whereAttrNum = 1;
        lineitemRel.whereIndex = (int *)host_mempool.alloc(sizeof(int) * 1);
        lineitemRel.outputIndex = (int *)host_mempool.alloc(sizeof(int) * 1);
        lineitemRel.outputNum = 1;
        lineitemRel.outputIndex[0] = 0;
        lineitemRel.whereIndex[0] = 1;
        lineitemRel.keepInGpu = 1;
        lineitemRel.filter = (struct whereCondition *)host_mempool.alloc(sizeof(struct whereCondition));
        (lineitemRel.filter)->nested = 0;
        (lineitemRel.filter)->expNum = 1;
        (lineitemRel.filter)->exp = (struct whereExp*)host_mempool.alloc(sizeof(struct whereExp) *1);
        (lineitemRel.filter)->andOr = EXP;
        (lineitemRel.filter)->exp[0].index    = 0;
        (lineitemRel.filter)->exp[0].relation = LTH_VEC;
        (lineitemRel.filter)->exp[0].dataPos  = lineitemTablePartial->dataPos[2];
        {
            memcpy((lineitemRel.filter)->exp[0].content, &(lineitemTablePartial->content[2]), sizeof(char *));
        }
        int dev_memsize = tableScanGPUMemSize(&lineitemRel);
        if(gpu_inner_mp.freesize() < dev_memsize)
            gpu_inner_mp.resize(gpu_inner_mp.usedsize() + dev_memsize);
        char *origin_pos = gpu_inner_mp.freepos();
        li0 = tableScan(&lineitemRel, &pp, &host_mempool, &gpu_inner_mp, NULL, NULL, NULL, NULL);
        gpu_inner_mp.freeto(origin_pos);

        clock_gettime(CLOCK_REALTIME, &diskStart);
        lineitemTablePartial->content[0] = NULL;
        lineitemTablePartial->content[1] = NULL;
        lineitemTablePartial->content[2] = NULL;
        freeScan(&lineitemRel, false);

        clock_gettime(CLOCK_REALTIME, &diskEnd);
        li0->colIdxNum = 0;
    }

    struct tableNode * gb_li0;
    printf("Process the group by node for gb_li0\n");
    if(li0->tupleNum == 0)
        gb_li0 = li0;
    else
    {

        struct groupByNode * gbNode = (struct groupByNode *)host_mempool.alloc(sizeof(struct groupByNode));
        gbNode->table = li0;
        gbNode->groupByColNum = 1;
        gbNode->groupByIndex = (int *)host_mempool.alloc(sizeof(int) * 1);
        gbNode->groupByType = (int *)host_mempool.alloc(sizeof(int) * 1);
        gbNode->groupBySize = (int *)host_mempool.alloc(sizeof(int) * 1);
        gbNode->groupByIndex[0] = 0;
        gbNode->groupByType[0] = gbNode->table->attrType[0];
        gbNode->groupBySize[0] = gbNode->table->attrSize[0];
        gbNode->outputAttrNum = 1;
        gbNode->attrType = (int *)host_mempool.alloc(sizeof(int) *1);
        gbNode->attrSize = (int *)host_mempool.alloc(sizeof(int) *1);
        gbNode->gbExp = (struct groupByExp *)host_mempool.alloc(sizeof(struct groupByExp) * 1);
        gbNode->tupleSize = 0;
        gbNode->attrType[0] = li0->attrType[0];
        gbNode->attrSize[0] = li0->attrSize[0];
        gbNode->tupleSize += li0->attrSize[0];
        gbNode->gbExp[0].func = NOOP;
        gbNode->gbExp[0].exp.op = NOOP;
        gbNode->gbExp[0].exp.exp = 0;
        gbNode->gbExp[0].exp.opNum = 1;
        gbNode->gbExp[0].exp.opType = COLUMN;
        gbNode->gbExp[0].exp.opValue = 0;

        printf("Before groupby, the tupleNum is %d\n", gbNode->table->tupleNum);
        printf("gpu_inner_mp.freesize() is %lu\n", gpu_inner_mp.freesize());
        int dev_memsize = groupByGPUMemSize(gbNode);
        printf("dev_memsize is %d\n", dev_memsize);
        if(gpu_inner_mp.freesize() < dev_memsize) {
            printf("usedsize + dev_memsize is %lu\n", gpu_inner_mp.usedsize() + dev_memsize);
            gpu_inner_mp.resize(gpu_inner_mp.usedsize() + dev_memsize);
        }
        char *origin_pos_groupby = gpu_inner_mp.freepos();
        gb_li0 = groupBy(gbNode, &pp, &host_mempool, &gpu_inner_mp);
        gpu_inner_mp.freeto(origin_pos_groupby);

        freeGroupByNode(gbNode, false);

    }

    // Process the SelectProjectNode for gb_li0
    printf("Process the select-project-node for sp_gb_li0\n");
    struct tableNode *sp_gb_li0;
    {
        char * subqRes0;
        char *free_pos;
        char *free_gpu_pos;
        struct tableNode *gb_li0SP;
        gb_li0SP = (struct tableNode *)host_mempool.alloc(sizeof(struct tableNode));
        gb_li0SP->totalAttr = 1;
        gb_li0SP->attrType = (int *)host_mempool.alloc(sizeof(int) * 1);
        gb_li0SP->attrSize = (int *)host_mempool.alloc(sizeof(int) * 1);
        gb_li0SP->attrIndex = (int *)host_mempool.alloc(sizeof(int) * 1);
        gb_li0SP->attrTotalSize = (int *)host_mempool.alloc(sizeof(int) * 1);
        gb_li0SP->dataPos = (int *)host_mempool.alloc(sizeof(int) * 1);
        gb_li0SP->dataFormat = (int *)host_mempool.alloc(sizeof(int) * 1);
        gb_li0SP->content = (char **)host_mempool.alloc(sizeof(char *) * 1);
        int tuple_size = 0;

        gb_li0SP->attrSize[0] = gb_li0->attrSize[0];
        gb_li0SP->attrIndex[0] = gb_li0->attrIndex[0];
        gb_li0SP->attrType[0] = gb_li0->attrType[0];
        gb_li0SP->dataPos[0] = gb_li0->dataPos[0];
        gb_li0SP->dataFormat[0] = gb_li0->dataFormat[0];
        gb_li0SP->attrTotalSize[0] = gb_li0->attrTotalSize[0];
        gb_li0SP->content[0] = gb_li0->content[0];
        tuple_size += gb_li0SP->attrSize[0];

        gb_li0SP->tupleSize = tuple_size;
        gb_li0SP->tupleNum = gb_li0->tupleNum;

        gb_li0SP->keepInGpuIdx = 1;
        sp_gb_li0 = gb_li0SP;
    }

    printf("passed after the select project node\n");
    // Join two tables: od0, sp_gb_li0
    struct tableNode *od0_sp_gb_li0;
    printf("Process the join-node for od0_sp_gb_li0\n");
    {

        char * subqRes0;
        char *free_pos;
        char *free_gpu_pos;
        struct joinNode jNode;
        jNode.leftTable = od0;
        jNode.rightTable = sp_gb_li0;
        jNode.totalAttr = 1;
        jNode.keepInGpu = (int *)host_mempool.alloc(sizeof(int) * 1);
        for(int k=0; k<1; k++)
            jNode.keepInGpu[k] = 1;
        jNode.leftOutputAttrNum = 1;
        jNode.rightOutputAttrNum = 0;
        jNode.leftOutputAttrType = (int *)host_mempool.alloc(sizeof(int)*1);
        jNode.leftOutputIndex = (int *)host_mempool.alloc(sizeof(int)*1);
        jNode.leftPos = (int *)host_mempool.alloc(sizeof(int)*1);
        jNode.tupleSize = 0;
        jNode.leftOutputIndex[0] = 0;
        jNode.leftOutputAttrType[0] = STRING;
        jNode.leftPos[0] = 0;
        jNode.tupleSize += od0->attrSize[0];
        jNode.rightOutputAttrType = (int *)host_mempool.alloc(sizeof(int)*0);
        jNode.rightOutputIndex = (int *)host_mempool.alloc(sizeof(int)*0);
        jNode.rightPos = (int *)host_mempool.alloc(sizeof(int)*0);
        jNode.leftKeyIndex = 1;
        jNode.rightKeyIndex = 0;
        struct tableNode *joinRes;
        int dev_memsize = hashJoinGPUMemSize(&jNode, false);
        if(gpu_inner_mp.freesize() < dev_memsize)
            gpu_inner_mp.resize(gpu_inner_mp.usedsize() + dev_memsize);
        char *origin_pos_join = gpu_inner_mp.freepos();
        joinRes = hashJoin(&jNode, &pp, &host_mempool, &gpu_inner_mp, NULL, NULL);
        gpu_inner_mp.freeto(origin_pos_join);

        od0_sp_gb_li0 = joinRes;
    }

    struct tableNode * gb_od0_sp_gb_li0;
    printf("Process the table-node for gb_od0_sp_gb_li0\n");
    if(od0_sp_gb_li0->tupleNum == 0)
        gb_od0_sp_gb_li0 = od0_sp_gb_li0;
    else
    {

        struct groupByNode * gbNode = (struct groupByNode *)host_mempool.alloc(sizeof(struct groupByNode));
        gbNode->table = od0_sp_gb_li0;
        gbNode->groupByColNum = 1;
        gbNode->groupByIndex = (int *)host_mempool.alloc(sizeof(int) * 1);
        gbNode->groupByType = (int *)host_mempool.alloc(sizeof(int) * 1);
        gbNode->groupBySize = (int *)host_mempool.alloc(sizeof(int) * 1);
        gbNode->groupByIndex[0] = 0;
        gbNode->groupByType[0] = gbNode->table->attrType[0];
        gbNode->groupBySize[0] = gbNode->table->attrSize[0];
        gbNode->outputAttrNum = 2;
        gbNode->attrType = (int *)host_mempool.alloc(sizeof(int) *2);
        gbNode->attrSize = (int *)host_mempool.alloc(sizeof(int) *2);
        gbNode->gbExp = (struct groupByExp *)host_mempool.alloc(sizeof(struct groupByExp) * 2);
        gbNode->tupleSize = 0;
        gbNode->attrType[0] = od0_sp_gb_li0->attrType[0];
        gbNode->attrSize[0] = od0_sp_gb_li0->attrSize[0];
        gbNode->tupleSize += od0_sp_gb_li0->attrSize[0];
        gbNode->gbExp[0].func = NOOP;
        gbNode->gbExp[0].exp.op = NOOP;
        gbNode->gbExp[0].exp.exp = 0;
        gbNode->gbExp[0].exp.opNum = 1;
        gbNode->gbExp[0].exp.opType = COLUMN;
        gbNode->gbExp[0].exp.opValue = 0;
        gbNode->tupleSize += sizeof(float);
        gbNode->attrType[1] = FLOAT;
        gbNode->attrSize[1] = sizeof(float);
        gbNode->gbExp[1].func = COUNT;
        gbNode->gbExp[1].exp.op = NOOP;
        gbNode->gbExp[1].exp.opNum = 1;
        gbNode->gbExp[1].exp.exp = 0;
        gbNode->gbExp[1].exp.opType = CONS;
*(float *)&(        gbNode->gbExp[1].exp.opValue) = 1;
        int dev_memsize = groupByGPUMemSize(gbNode);
        if(gpu_inner_mp.freesize() < dev_memsize)
            gpu_inner_mp.resize(gpu_inner_mp.usedsize() + dev_memsize);
        char *origin_pos_groupby = gpu_inner_mp.freepos();
        gb_od0_sp_gb_li0 = groupBy(gbNode, &pp, &host_mempool, &gpu_inner_mp);
        gpu_inner_mp.freeto(origin_pos_groupby);

        freeGroupByNode(gbNode, false);

    }

    result = gb_od0_sp_gb_li0;
    struct materializeNode mn;
    mn.table = result;
    char *final = materializeCol(&mn, &pp);
    printMaterializedTable(mn, final);


    clock_gettime(CLOCK_REALTIME, &end);
    double timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
    printf("<--Disk Load Time-->           : %lf\n", diskTotal/(1000*1000));
    printf("\n");
    printf("<--Build index time-->         : %lf\n", pp.buildIndexTotal/(1000*1000));

    printf("\n");
    printf("<---SUB()--->\n");
    printf("Outer table size           : %d\n", pp.outerTableSize);

    printf("Result table size          : %d\n", pp.resultTableSize);

    printf("<----------------->");
    printf("\n");
    printf("<---TableScan()--->\n");
    printf("Total time      : %lf\n", pp.tableScanTotal/(1000*1000));
    printf("Calls           : %d\n", pp.tableScanCount);

    printf("Step 1 - memCopy where clause                 : %lf\n", pp.whereMemCopy_s1/(1000*1000));
    printf("Step 2 - memCopy predicate col                : %lf\n", pp.dataMemCopy_s2/(1000*1000));
    printf("Step 3 - Scan                                 : %lf\n", pp.scanTotal_s3/(1000*1000));
    printf("Idx Step 3.1 - Get index position             : %lf\n", pp.getIndexPos_idxS1/(1000*1000));
    printf("Idx Step 3.2 - Get range                      : %lf\n", pp.getRange_idxS2/(1000*1000));
    printf("Idx Step 3.3 - Convert addrs to elements      : %lf\n", pp.convertMemToElement_idxS3/(1000*1000));
    printf("Idx Step 3.4 - Get mapping position           : %lf\n", pp.getMapping_idxS4/(1000*1000));
    printf("Idx Step 3.5 - Set bitmap to zero             : %lf\n", pp.setBitmapZeros_idxS5/(1000*1000));
    printf("Idx Step 3.6 - Build bitmap                   : %lf\n", pp.buildBitmap_idxS6/(1000*1000));
    printf("Step 4 - CountRes(PreScan)                    : %lf\n", pp.preScanTotal_s4/(1000*1000));
    printf("PreScan Step 4.1 - Count selected rows kernel : %lf\n", pp.countScanKernel_countS1/(1000*1000));
    printf("PreScan Step 4.2 - scanImpl time              : %lf\n", pp.scanImpl_countS2/(1000*1000));
    printf("scanImpl Step 4.2.1 - preallocBlockSums time  : %lf\n", pp.preallocBlockSums_scanImpl_S1/(1000*1000));
    printf("scanImpl Step 4.2.2 - prescanArray time       : %lf\n", pp.prescanArray_scanImpl_S2/(1000*1000));
    printf("scanImpl Step 4.2.3 - deallocBlockSums time   : %lf\n", pp.deallocBlockSums_scanImpl_S3/(1000*1000));
    printf("prescan Step 4.2.3.1 - set variables time     : %lf\n", pp.setVar_prescan_S1/(1000*1000));
    printf("prescan Step 4.2.3.2 - prescan Kernel time    : %lf\n", pp.preScanKernel_prescan_S2/(1000*1000));
    printf("prescan Step 4.2.3.3 - uniformAdd Kernel time : %lf\n", pp.uniformAddKernel_prescan_S3/(1000*1000));
    printf("Step 5 - memReturn countRes                   : %lf\n", pp.preScanResultMemCopy_s5/(1000*1000));
    printf("Step 6 - Copy rest of columns                 : %lf\n", pp.dataMemCopyOther_s6/(1000*1000));
    printf("Step 7 - Materialize result                   : %lf\n", pp.materializeResult_s7/(1000*1000));
    printf("Step 8 - Copy final result                    : %lf\n", pp.finalResultMemCopy_s8/(1000*1000));
    printf("Other 1 - Create tableNode                    : %lf\n", pp.create_tableNode_S01/(1000*1000));
    printf("Other 2 - Malloc res                          : %lf\n", pp.mallocRes_S02/(1000*1000));
    printf("Other 3 - Deallocate buffers                  : %lf\n", pp.deallocateBuffs_S03/(1000*1000));
    printf("<----------------->");
    printf("\n");
    printf("Total Time: %lf\n", timeE/(1000*1000));
}

