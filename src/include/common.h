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

#ifndef __SSB_COMMON__
#define __SSB_COMMON__

#define BILLION     1000000000
#define BLOCKNUM    (100*1024*1024)
#define HSIZE 2891072
#define FLOAT_MAX 0x7f800000
#define FLOAT_MIN 0xff800000

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)

#define NP2(n)              do {                    \
    n--;                                            \
    n |= n >> 1;                                    \
    n |= n >> 2;                                    \
    n |= n >> 4;                                    \
    n |= n >> 8;                                    \
    n |= n >> 16;                                   \
    n ++; } while (0)


enum {

/* data format */
    RLE = 0,
    DICT,
    DELTA,
    UNCOMPRESSED,

/* data type supported in schema */
    INT,
    FLOAT,
    STRING,
    DATE,

/* supported relation in exp */
    EQ,
    GTH,
    LTH,
    GEQ,
    LEQ,
    NOT_EQ,
    IN,
    NOT_IN,
    LIKE,
    NOT_LIKE,

    EQ_VEC,
    GTH_VEC,
    LTH_VEC,
    GEQ_VEC,
    LEQ_VEC,
    NOT_EQ_VEC,
    IN_VEC,
    NOT_IN_VEC,
    EXISTS,
    NOT_EXISTS_VEC,

/* for where condition */
    AND ,
    OR,
    EXP,
    EXPSUB,                 /* the where exp is a math exp, and the column is correlated */

/* supported groupby function */
    MIN,
    MAX,
    COUNT,
    SUM,
    AVG,

/* supported math operation */
    PLUS,
    MINUS,
    MULTIPLY,
    DIVIDE,

/* op type for mathExp */
    COLUMN,
    COLUMN_DECIMAL,
    COLUMN_INTEGER,
    CONS,

/* data position */
    GPU,
    MEM,
    PINNED,
    UVA,
    MMAP,
    DISK,
    TMPFILE,

/* order by sequence */
    ASC,
    DESC,

    NOOP
};

/* header of each block in the column */

struct columnHeader{
    long totalTupleNum; /* the total number of tuples in this column */
    long tupleNum;      /* the number of tuples in this block */
    long blockSize;     /* the size of the block in bytes */
    int blockTotal;     /* the total number of blocks that this column is divided into */
    int blockId;        /* the block id of the current block */
    int format;         /* the format of the current block */
    char padding[4060]; /* for futher use */
};

/*
 * if the column is compressed using dictionary encoding method,
 * distHeader will come behind the columnHeader.
 * The size of the dictHeader varies depending on the size of the dictionary.
 */

#define MAX_DICT_NUM    30000

struct dictHeader{
    int dictNum;                /* number of distict values in the dictionary */
    int bitNum;                 /* the number of bits used to compress the data */
    int hash[MAX_DICT_NUM];     /* the hash table to store the dictionaries */
};

struct rleHeader{
    int dictNum;
};

struct whereExp{
    int index;
    int relation;
    char content[32];
    int dataPos;
    int vlen;
};

struct whereCondition{
    int andOr;
    int nested;
    int nestedRel;
    int expNum;
    struct whereExp *exp;
    struct whereCondition * con;
};

struct scanNode{
    struct tableNode *tn ;          /* the tableNode to be scanned */
    int hasWhere;                   /* whether the node has where condition */
    int outputNum;                  /* the number of attributes that will be projected */
    int *outputIndex;               /* the index of projected attributes in the tableNode */
    int whereAttrNum;               /* the number of attributes in the where condition */
    int * whereIndex;               /* the index of each col in the table */
    struct whereCondition * filter; /* the where conditioin */
    int keepInGpu;                  /* whether all the results should be kept in GPU memory or not */

};

/*
 * For dedup, we currently only support integers
 */

struct dedupNode{
    struct tableNode *tn;
    int index;                      /* the index of the column that needs to be deduped*/
};

struct vecNode{
    struct tableNode *tn;
    int index;
};

struct mathExp {
    int op;             /* the math operation */
    int opNum;          /* the number of operands */

    long exp;           /* if the opNum is 2, this field stores pointer that points to the two operands whose type is mathExp */

/* when opNum is 1 */
    int opType;         /* whether it is a regular column or a constant */
    int opValue;        /* it is the index of the column or the value of the constant */
};

struct tableNode{
    int totalAttr;          /* the total number of attributes */
    long tupleNum;          /* the number of tuples in the relation */
    int tupleSize;          /* the size of each tuple */
    int * attrType;         /* the type of each attributes */
    int * attrSize;         /* the size of each attributes */
    int * attrTotalSize;    /* the total size of each attribute */
    int * attrIndex;        /* the index of each attribute in the table */
    char **content;         /* the actual content of each attribute, organized by columns */
    int * dataPos;          /* the position of the data, whether in disk, memory or GPU global memory */
    int * dataFormat;       /* the format of each column */

    /* Indexing */
    int colIdxNum;          /* the total number of sorted indexed columns */
    int *colIdx;            /* the list with all the sorted indexed columns */
    int **posIdx;           /* the possition of each value in the original table (col1, col2 etc) */
    int **contentIdx;       /* the values sorted (col1, col2 etc) - ASSUMPTION, support only int for indexing */
    int keepInGpuIdx;       /* indicates if the index should be kept in the device or not */
    int indexPos;           /* the position of the index, whether in disk, memory or GPU global memory */
};

struct groupByExp{
    int func;               /* the group by function */
    struct mathExp exp;     /* the math exp */
};

struct groupByNode{
    struct tableNode * table;   /* the table node to be grouped by */

    int groupByColNum;          /* the number of columns that will be grouped on */
    int * groupByIndex;         /* the index of the columns that will be grouped on, -1 means group by a constant */
    int * groupByType;          /* the type of the group by column */
    int * groupBySize;          /* the size of the group by column */

    int outputAttrNum;          /* the number of output attributes */
    int *attrType;              /* the type of each output attribute */
    int *attrSize;              /* the size of each output attribute */
    struct groupByExp * gbExp;  /* the group by expression */

    int tupleSize;              /* the size of the tuple in the join result */

    int * keepInGpu;            /* whether the results should be kept in gpu */

};

struct sortRecord{
    unsigned int key;           /* the key to be sorted */
    unsigned int pos;           /* the position of the corresponding record */
};

struct orderByNode{
    struct tableNode * table;   /* the table node to be ordered by */

    int orderByNum;             /* the number of columns that will be ordered on */
    int *orderBySeq;            /* asc or desc */
    int *orderByIndex;          /* the index of each order by column in the table */

};

struct materializeNode{
    struct tableNode * table;   /* the table node to be materialized */
};

struct statistic{
    float kernel;
    float pcie;
    float total;
    
    // ========= tableScan() =========

    // Indexing profiling
    double buildIndexTotal;
    
    // Total scan time prof
    double tableScanTotal;
    int tableScanCount;

    // Step 1 - Copy where clause
    double whereMemCopy_s1;

    // Step 2 - Copy data
    double dataMemCopy_s2;

    // Step 3 - Scan
    double scanTotal_s3;

    // Step 4 - Count result (PreScan)
    double preScanTotal_s4;
    int preScanCount_s4;

    // Step 5 - Return count result (PreScan result)
    double preScanResultMemCopy_s5;

    // Step 6 - Copy to device all other columns  (to materialize the final result)
    double dataMemCopyOther_s6;

    // Step 7 - Materialize result
    double materializeResult_s7;

    // Step 8 - Copy final result
    double finalResultMemCopy_s8;

    // Other - 01 (tableScan) Time on other stuff inside tableScan()
    double create_tableNode_S01;

    // Other - 02 (allocate result) Time on other stuff inside tableScan()
    double mallocRes_S02;

    // Other - 02 (deallocate bufs) Time on other stuff inside tableScan()
    double deallocateBuffs_S03;

    // Index Step 3.1 - Get index position 
    double getIndexPos_idxS1;

    // Index Step 3.2 - Get range
    double getRange_idxS2;

    // Index Step 3.3 - Convert mem addrs to elements
    double convertMemToElement_idxS3;

    // Index Step 3.4 - Get mapping position 
    double getMapping_idxS4;

    // Index Step 3.5 - Set bitmap to zero
    double setBitmapZeros_idxS5;

    // Index Step 3.6 - Build bitmap
    double buildBitmap_idxS6;

    // Count Step 4.1 - Count selected rows kernel
    double countScanKernel_countS1;

    // Count Step 4.2 - scanImpl time
    double scanImpl_countS2;

    // scanImpl Step 4.2.1 - preallocBlockSums time
    double preallocBlockSums_scanImpl_S1;

    // scanImpl Step 4.2.2 - prescanArray time
    double prescanArray_scanImpl_S2;

    // scanImpl Step 4.2.3 - deallocBlockSums time
    double deallocBlockSums_scanImpl_S3;

    // prescan Step 4.2.3.1 - set variables time
    double setVar_prescan_S1;

    // prescan Step 4.2.3.2 - prescan Kernel time
    double preScanKernel_prescan_S2;

    // prescan Step 4.2.3.3 - uniformAdd Kernel time
    double uniformAddKernel_prescan_S3;

    // ===============================


    // ========= hashJoin() =========

    double join_totalTime;
    int join_callTimes;
    int join_leftTableSize;
    int join_rightTableSize;

    // Step 1 - Allocate memory for intermediate results
    double joinProf_step1_allocateMem;

    // Step 2 - Build hashTable
    double joinProf_step2_buildHash;

    // Step 2.1 - Allocate memory
    double joinProf_step21_allocateMem;

    // Step 2.2 - Count_hash_num
    double joinProf_step22_Count_hash_num;

    // Step 2.3 - scanImpl
    double joinProf_step23_scanImpl;
    
    // Step 2.4 - build_hash_table (+ memCopy op)
    double joinProf_step24_buildhash_kernel_memcopy;

    // Step 3 - Join
    double joinProf_step3_join;

    //Step 3.1 - Allocate memory
    double joinProf_step31_allocateMem;

    //Step 3.2 - Exclusive scan
    double joinProf_step32_exclusiveScan;  

    //Step 3.3 - Prob and memcpy ops
    double joinProf_step33_prob; 

    // Step 4 - De-allocate memory
    double joinProf_step4_deallocate;
    // ===============================


    // ========= groupBy() =========

    double groupby_totalTime;
    int groupby_callTimes;

    //Step 1 - Allocate memory for intermediate results
    double groupby_step1_allocMem;

    //Step 2 - Copy data to GPU
    double groupby_step2_copyToDevice;
    
    //Step 3 - build_groupby_key kernel
    double groupby_step3_buildGroupByKey;

    //Step 4 - Count number of groups
    double groupby_step4_groupCount;

    //Step 5 - Allocate memory for result
    double groupby_step5_AllocRes;

    //Step 6 - Copy columns to device
    double groupby_step6_copyDataCols;

    //Step 7 - Calculate aggregate values
    double groupby_step7_computeAgg;

    //Step 8 - De-allocate memory
    double groupby_step8_deallocate;
    // ===============================

    // ============ SUB() ============
    int outerTableSize;
    int resultTableSize;
    // ===============================

};


#endif
