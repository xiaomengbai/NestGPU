
/*
 * tableScan Prerequisites:
 *  1. the input data can be fit into GPU device memory
 *  2. input data are stored in host memory
 *
 * Input:
 *  sn: contains the data to be scanned and the predicate information
 *  rslt: object to store the result
 *  pp: records statistics such kernel execution time and PCIe transfer time
 *
 * Output:
 *  A new table node
 */
 struct tableNode * tableScanNest(struct scanNode *sn, struct tableNode *rslt, struct statistic *pp,

    //Other objs
    char ** column,
    int * whereFree,
    int * colWherePos,

    //Gpu objs
    int * gpuCount,
    int * gpuFilter,
    int * gpuPsum,
    dim3 grid,
    dim3 block

){

    //Start timer (total tableScan())
    struct timespec startTableScanTotal,endTableScanTotal;
    clock_gettime(CLOCK_REALTIME,&startTableScanTotal);

    //Start timer for Other - 01 (tableScan)
    struct timespec startS01,endS01;
    clock_gettime(CLOCK_REALTIME,&startS01);

    //Set tableNode result already given by the user
    struct tableNode *res = rslt;

    //Some other info
    int tupleSize = 0;
    long totalTupleNum = sn->tn->tupleNum;
    int threadNum = grid.x * block.x;
    int attrNum = sn->whereAttrNum;
    int blockNum = totalTupleNum / block.x + 1;
    int count;


    assert(sn->hasWhere !=0);
    assert(sn->filter != NULL);

    struct whereCondition *where = sn->filter;

    //Stop timer for Other - 01 (tableScan)
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS01);
    pp->create_tableNode_S01 += (endS01.tv_sec - startS01.tv_sec)* BILLION + endS01.tv_nsec - startS01.tv_nsec;

    //Keeps index col pos
    int idxPos = -1;

    /*
     * The first step is to evaluate the selection predicates and generate a vetor to form the final results.
    */
    if(1){

        //Start timer for Step 1 - Copy where clause 
        struct timespec startS1, endS1;
        clock_gettime(CLOCK_REALTIME,&startS1);

        struct whereExp * gpuExp = NULL;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuExp, sizeof(struct whereExp)));
        //CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[0], sizeof(struct whereExp), cudaMemcpyHostToDevice));

        /*
         * Currently we evaluate the predicate one by one.
         * When consecutive predicates are accessing the same column, we don't release the GPU device memory
         * that store the accessed column until all these predicates have been evaluated. Otherwise the device
         * memory will be released immediately after the corresponding predicate has been evaluated.
         *
         * (@whereIndex, @prevWhere), (@index,  @prevIndex), (@format, @prevFormat) are used to decide
         * whether two consecutive predicates access the same column with the same format.
         */

        int whereIndex = where->exp[0].index;
        int index = sn->whereIndex[whereIndex];
        int prevWhere = whereIndex;
        int prevIndex = index;

        int format = sn->tn->dataFormat[index];
        int prevFormat = format;

        if( (where->exp[0].relation == EQ_VEC || where->exp[0].relation == NOT_EQ_VEC || where->exp[0].relation == GTH_VEC || where->exp[0].relation == LTH_VEC || where->exp[0].relation == GEQ_VEC || where->exp[0].relation == LEQ_VEC) &&
            (where->exp[0].dataPos == MEM || where->exp[0].dataPos == MMAP || where->exp[0].dataPos == PINNED) )
        {
            char vec_addr_g[32];
            size_t vec_size = sn->tn->attrSize[index] * sn->tn->tupleNum;
            CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **) vec_addr_g, vec_size) );
            CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(*((void **)vec_addr_g), *((void **) where->exp[0].content), vec_size, cudaMemcpyHostToDevice) );

            /* free memory here? */
            free( *((void **) where->exp[0].content) );
            memcpy(where->exp[0].content, vec_addr_g, 32);
            where->exp[0].dataPos = GPU;
        }

        if( (where->exp[0].relation == IN || where->exp[0].relation == LIKE) &&
            (where->exp[0].dataPos == MEM || where->exp[0].dataPos == MMAP || where->exp[0].dataPos == PINNED) )
        {
            char vec_addr_g[32];
            size_t vec_size = sn->tn->attrSize[index] * where->exp[0].vlen;
            CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **) vec_addr_g, vec_size) );
            CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(*((void **)vec_addr_g), *((void **) where->exp[0].content), vec_size, cudaMemcpyHostToDevice) );

            free( *((void **) where->exp[0].content) );
            memcpy(where->exp[0].content, vec_addr_g, 32);
            where->exp[0].dataPos = GPU;
        }

        if( where->exp[0].relation == IN_VEC &&
            (where->exp[0].dataPos == MEM || where->exp[0].dataPos == MMAP || where->exp[0].dataPos == PINNED) )
        {
            char vec_addr_g[32];
            size_t vec1_size = sizeof(void *) * sn->tn->tupleNum;
            void **vec1_addrs = (void **)malloc(vec1_size);
            for(int i = 0; i < sn->tn->tupleNum; i++) {
                // [int: vec_len][char *: string1][char *: string2] ...
                size_t vec2_size = *((*(int ***)(where->exp[0].content))[i]) * sn->tn->attrSize[index] + sizeof(int);
                CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **) &vec1_addrs[i], vec2_size) );
                CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy( vec1_addrs[i], (*(char ***)where->exp[0].content)[i], vec2_size, cudaMemcpyHostToDevice) );
                free( (*(char ***)where->exp[0].content)[i] );
            }
            CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **) vec_addr_g, vec1_size) );
            CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(*((void **)vec_addr_g), vec1_addrs, vec1_size, cudaMemcpyHostToDevice) );
            free(vec1_addrs);

            free(*(char ***)where->exp[0].content);
            memcpy(where->exp[0].content, vec_addr_g, 32);
            where->exp[0].dataPos = GPU;
        }

        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[0], sizeof(struct whereExp), cudaMemcpyHostToDevice));

        //Stop for Step 1 - Copy where clause 
        CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
        clock_gettime(CLOCK_REALTIME, &endS1);
        pp->whereMemCopy_s1 += (endS1.tv_sec - startS1.tv_sec)* BILLION + endS1.tv_nsec - startS1.tv_nsec;
                 
       /*   
         * @dNum, @byteNum and @gpuDictFilter are for predicates that need to access dictionary-compressed columns.
         */
        int dNum;
        int byteNum;
        int *gpuDictFilter = NULL;

        /*
         * We will allocate GPU device memory for a column if it is stored in the host pageable or pinned memory.
         * If it is configured to utilize the UVA technique, no GPU device memory will be allocated.
         */

        if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[whereIndex], sn->tn->attrTotalSize[index]));

        if(format == UNCOMPRESSED){
            
            //Start timer for Step 2 - Copy data
            struct timespec startS2, endS2;
            clock_gettime(CLOCK_REALTIME,&startS2);

            if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
            else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                column[whereIndex] = sn->tn->content[index];

            //Stop timer for Step 2 - Copy data
            CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
            clock_gettime(CLOCK_REALTIME, &endS2);
            pp->dataMemCopy_s2 += (endS2.tv_sec - startS2.tv_sec)* BILLION + endS2.tv_nsec - startS2.tv_nsec;
                
            int rel = where->exp[0].relation;
            if(sn->tn->attrType[index] == INT){
                int whereValue;
                int *whereVec;
                if(rel == EQ || rel == NOT_EQ || rel == GTH || rel == LTH || rel == GEQ || rel == LEQ)
                    whereValue = *((int*) where->exp[0].content);
                else
                    whereVec = *((int **) where->exp[0].content);

                if(rel==EQ){
                    //Check if this column is indexed
                    if (sn->tn->colIdxNum != 0){
                        for (int k=0; k<sn->tn->colIdxNum; k++){
                            if (sn->tn->colIdx[k] == index){
                                idxPos = k; 
                            }
                        }
                    }

                    //Start timer for Step 3 - Scan
                    struct timespec startS3, endS3;
                    clock_gettime(CLOCK_REALTIME,&startS3);
                    
                    if (idxPos >= 0){ 

                        //Regular scan
                        //genScanFilter_init_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);

                        //Index scan!
                        res->tupleNum = indexScanInt(sn->tn, index, idxPos, whereValue, gpuFilter, pp);
                        
                        /* DEBUG CODE - DO NOT REMOVE FOR NOW */
                        // int* originalRes = (int *)malloc(sizeof(int) * totalTupleNum);
                        // CHECK_POINTER(originalRes); 
                        // CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(originalRes, gpuFilter, sizeof(int) * totalTupleNum, cudaMemcpyDeviceToHost));
                        // printf("This is what is being returned by serial scan:\n");
                        // for (int i=0;i<totalTupleNum;i++){
                        //     if (originalRes[i] != 0 ){
                        //         printf ("Value[%d]: %d \n",i,originalRes[i]);
                        //     }
                        // }
                        // free(originalRes);

                    }else{
                        genScanFilter_init_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                    }

                    //Stop timer for Step 3 - Scan
                    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
                    clock_gettime(CLOCK_REALTIME, &endS3);
                    pp->scanTotal_s3 += (endS3.tv_sec - startS3.tv_sec)* BILLION + endS3.tv_nsec - startS3.tv_nsec;

                }else if(rel == NOT_EQ)
                    genScanFilter_init_int_neq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == GTH)
                    genScanFilter_init_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == LTH)
                    genScanFilter_init_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == GEQ)
                    genScanFilter_init_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if (rel == LEQ)
                    genScanFilter_init_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if (rel == EQ_VEC)
                    genScanFilter_init_int_eq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if (rel == NOT_EQ_VEC)
                    genScanFilter_init_int_neq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if (rel == GTH_VEC)
                    genScanFilter_init_int_gth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if (rel == LTH_VEC)
                    genScanFilter_init_int_lth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if (rel == GEQ_VEC)
                    genScanFilter_init_int_geq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if (rel == LEQ_VEC)
                    genScanFilter_init_int_leq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);

            }else if (sn->tn->attrType[index] == FLOAT){
                float whereValue, *whereVec;
                if(rel == EQ || rel == NOT_EQ || rel == GTH || rel == LTH || rel == GEQ || rel == LEQ)
                    whereValue = *((float*) where->exp[0].content);
                else
                    whereVec = *((float **) where->exp[0].content);

                if(rel==EQ)
                    genScanFilter_init_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel==NOT_EQ)
                    genScanFilter_init_float_neq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == GTH)
                    genScanFilter_init_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == LTH)
                    genScanFilter_init_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel == GEQ)
                    genScanFilter_init_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if (rel == LEQ)
                    genScanFilter_init_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                else if(rel==EQ_VEC)
                    genScanFilter_init_float_eq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if(rel==NOT_EQ_VEC)
                    genScanFilter_init_float_neq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if(rel == GTH_VEC)
                    genScanFilter_init_float_gth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if(rel == LTH_VEC)
                    genScanFilter_init_float_lth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if(rel == GEQ_VEC)
                    genScanFilter_init_float_geq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                else if (rel == LEQ_VEC)
                    genScanFilter_init_float_leq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);


            }else{
                if(rel == EQ)
                    genScanFilter_init_eq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if(rel == NOT_EQ)
                    genScanFilter_init_neq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == GTH)
                    genScanFilter_init_gth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == LTH)
                    genScanFilter_init_lth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == GEQ)
                    genScanFilter_init_geq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == LEQ)
                    genScanFilter_init_leq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if(rel == EQ_VEC)
                    genScanFilter_init_eq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if(rel == NOT_EQ_VEC)
                    genScanFilter_init_neq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == GTH_VEC)
                    genScanFilter_init_gth_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == LTH_VEC)
                    genScanFilter_init_lth_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == GEQ_VEC)
                    genScanFilter_init_geq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == LEQ_VEC)
                    genScanFilter_init_leq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                else if (rel == IN)
                    genScanFilter_init_in<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
                else if (rel == IN_VEC)
                    genScanFilter_init_in_vec<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
                else if (rel == LIKE)
                    genScanFilter_init_like<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
            }

        }else if(format == DICT){

            /*Data are stored in the host memory for selection*/

            struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
            dNum = dheader->dictNum;
            byteNum = dheader->bitNum/8;

            struct dictHeader* gpuDictHeader = NULL;
            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

            if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
            else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                column[whereIndex] = sn->tn->content[index];

            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

            genScanFilter_dict_init<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);

            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

        }else if(format == RLE){

            if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
            else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                column[whereIndex] = sn->tn->content[index];

            genScanFilter_rle<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, where->andOr, gpuFilter);
        }

        int dictFilter = 0;
        int dictFinal = OR;
        int dictInit = 1;

        //Start timer for Step 1 - Copy where clause (Part2)
        struct timespec startS12, endS12;
        clock_gettime(CLOCK_REALTIME,&startS12);

        for(int i=1;i<where->expNum;i++){
            whereIndex = where->exp[i].index;
            index = sn->whereIndex[whereIndex];
            format = sn->tn->dataFormat[index];

            if( (where->exp[i].relation == EQ_VEC || where->exp[i].relation == NOT_EQ_VEC || where->exp[i].relation == GTH_VEC || where->exp[i].relation == LTH_VEC || where->exp[i].relation == GEQ_VEC || where->exp[i].relation == LEQ_VEC) &&
                (where->exp[i].dataPos == MEM || where->exp[i].dataPos == MMAP || where->exp[i].dataPos == PINNED) )
            {
                char vec_addr_g[32];
                size_t vec_size = sn->tn->attrSize[index] * sn->tn->tupleNum;
                CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **) vec_addr_g, vec_size) );
                CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(*((void **)vec_addr_g), *((void **) where->exp[i].content), vec_size, cudaMemcpyHostToDevice) );

                free( *((void **) where->exp[i].content) );
                memcpy(where->exp[i].content, vec_addr_g, 32);
                where->exp[i].dataPos = GPU;
            }

            if( (where->exp[i].relation == IN || where->exp[i].relation == LIKE) &&
                (where->exp[i].dataPos == MEM || where->exp[i].dataPos == MMAP || where->exp[i].dataPos == PINNED) )
            {
                char vec_addr_g[32];
                size_t vec_size = sn->tn->attrSize[index] * where->exp[i].vlen;
                CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **) vec_addr_g, vec_size) );
                CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(*((void **)vec_addr_g), *((void **) where->exp[i].content), vec_size, cudaMemcpyHostToDevice) );

                free( *((void **) where->exp[i].content) );
                memcpy(where->exp[i].content, vec_addr_g, 32);
                where->exp[i].dataPos = GPU;
            }

            if( where->exp[i].relation == IN_VEC &&
                (where->exp[i].dataPos == MEM || where->exp[i].dataPos == MMAP || where->exp[i].dataPos == PINNED) )
            {
                char vec_addr_g[32];
                size_t vec1_size = sizeof(void *) * sn->tn->tupleNum;
                void **vec1_addrs = (void **)malloc(vec1_size);
                for(int j = 0; j < sn->tn->tupleNum; j++) {
                    // [int: vec_len][char *: string1][char *: string2] ...
                    size_t vec2_size = *((*(int ***)(where->exp[i].content))[j]) * sn->tn->attrSize[index] + sizeof(int);
                    CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **) &vec1_addrs[j], vec2_size) );
                    CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy( vec1_addrs[j], (*(char ***)where->exp[i].content)[j], vec2_size, cudaMemcpyHostToDevice) );
                    free( (*(char ***)where->exp[i].content)[j] );
                }
                CUDA_SAFE_CALL_NO_SYNC( cudaMalloc((void **) vec_addr_g, vec1_size) );
                CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(*((void **)vec_addr_g), vec1_addrs, vec1_size, cudaMemcpyHostToDevice) );
                free(vec1_addrs);

                free(*(char ***)where->exp[i].content);
                memcpy(where->exp[i].content, vec_addr_g, 32);
                where->exp[i].dataPos = GPU;
            }


            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuExp, &where->exp[i], sizeof(struct whereExp), cudaMemcpyHostToDevice));

            if(prevIndex != index){

                 /*When the two consecutive predicates access different columns*/

                if(prevFormat == DICT){
                    if(dictInit == 1){
                        transform_dict_filter_init<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);
                        dictInit = 0;
                    }else if(dictFinal == OR)
                        transform_dict_filter_or<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);
                    else
                        transform_dict_filter_and<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter,byteNum);

                    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
                    dictFinal = where->andOr;
                }

                if(whereFree[prevWhere] == 1 && (sn->tn->dataPos[prevIndex] == MEM || sn->tn->dataPos[prevIndex] == MMAP || sn->tn->dataPos[prevIndex] == PINNED))
                    CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[prevWhere]));

                if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &column[whereIndex] , sn->tn->attrTotalSize[index]));

                if(format == DICT){
                    if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
                    else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                        column[whereIndex] = sn->tn->content[index];

                    struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
                    dNum = dheader->dictNum;
                    byteNum = dheader->bitNum/8;

                    struct dictHeader * gpuDictHeader = NULL;
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));
                    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

                    genScanFilter_dict_init<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
                    dictFilter= -1;
                    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

                }else{
                    if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(column[whereIndex], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
                    else if (sn->tn->dataPos[index] == UVA || sn->tn->dataPos[index] == GPU)
                        column[whereIndex] = sn->tn->content[index];
                }

                prevIndex = index;
                prevWhere = whereIndex;
                prevFormat = format;
            }


            if(format == UNCOMPRESSED){
                int rel = where->exp[i].relation;
                if(sn->tn->attrType[index] == INT){
                    int whereValue;
                    int *whereVec;
                    if(rel == EQ || rel == NOT_EQ || rel == GTH || rel == LTH || rel == GEQ || rel == LEQ)
                        whereValue = *((int*) where->exp[i].content);
                    else
                        whereVec = *((int **) where->exp[i].content);

                    if(where->andOr == AND){
                        if(rel==EQ)
                            genScanFilter_and_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel==NOT_EQ)
                            genScanFilter_and_int_neq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GTH)
                            genScanFilter_and_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == LTH)
                            genScanFilter_and_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GEQ)
                            genScanFilter_and_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_and_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel==EQ_VEC)
                            genScanFilter_and_int_eq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel==NOT_EQ_VEC)
                            genScanFilter_and_int_neq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == GTH_VEC)
                            genScanFilter_and_int_gth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == LTH_VEC)
                            genScanFilter_and_int_lth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == GEQ_VEC)
                            genScanFilter_and_int_geq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if (rel == LEQ_VEC)
                            genScanFilter_and_int_leq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                    }else{
                        if(rel==EQ)
                            genScanFilter_or_int_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel==NOT_EQ)
                            genScanFilter_or_int_neq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GTH)
                            genScanFilter_or_int_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == LTH)
                            genScanFilter_or_int_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GEQ)
                            genScanFilter_or_int_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_or_int_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel==EQ_VEC)
                            genScanFilter_or_int_eq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel==NOT_EQ_VEC)
                            genScanFilter_or_int_neq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == GTH_VEC)
                            genScanFilter_or_int_gth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == LTH_VEC)
                            genScanFilter_or_int_lth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == GEQ_VEC)
                            genScanFilter_or_int_geq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if (rel == LEQ_VEC)
                            genScanFilter_or_int_leq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);

                    }

                } else if (sn->tn->attrType[index] == FLOAT){
                    float whereValue, *whereVec;
                    if(rel == EQ || rel == NOT_EQ || rel == GTH || rel == LTH || rel == GEQ || rel == LEQ)
                        whereValue = *((float*) where->exp[i].content);
                    else
                        whereVec = *((float**) where->exp[i].content);

                    if(where->andOr == AND){
                        if(rel==EQ)
                            genScanFilter_and_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel==NOT_EQ)
                            genScanFilter_and_float_neq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GTH)
                            genScanFilter_and_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == LTH)
                            genScanFilter_and_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GEQ)
                            genScanFilter_and_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_and_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel==EQ_VEC)
                            genScanFilter_and_float_eq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel==NOT_EQ_VEC)
                            genScanFilter_and_float_neq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);    
                        else if(rel == GTH_VEC)
                            genScanFilter_and_float_gth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == LTH_VEC)
                            genScanFilter_and_float_lth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == GEQ_VEC)
                            genScanFilter_and_float_geq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if (rel == LEQ_VEC)
                            genScanFilter_and_float_leq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);

                    }else{
                        if(rel==EQ)
                            genScanFilter_or_float_eq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel==NOT_EQ)
                            genScanFilter_or_float_neq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GTH)
                            genScanFilter_or_float_gth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == LTH)
                            genScanFilter_or_float_lth<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel == GEQ)
                            genScanFilter_or_float_geq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_or_float_leq<<<grid,block>>>(column[whereIndex],totalTupleNum, whereValue, gpuFilter);
                        else if(rel==EQ_VEC)
                            genScanFilter_or_float_eq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel==NOT_EQ_VEC)
                            genScanFilter_or_float_neq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == GTH_VEC)
                            genScanFilter_or_float_gth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == LTH_VEC)
                            genScanFilter_or_float_lth_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if(rel == GEQ_VEC)
                            genScanFilter_or_float_geq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);
                        else if (rel == LEQ_VEC)
                            genScanFilter_or_float_leq_vec<<<grid,block>>>(column[whereIndex],totalTupleNum, whereVec, gpuFilter);

                    }
                }else{
                    if(where->andOr == AND){
                        if (rel == EQ)
                            genScanFilter_and_eq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == NOT_EQ)
                            genScanFilter_and_neq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GTH)
                            genScanFilter_and_gth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LTH)
                            genScanFilter_and_lth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GEQ)
                            genScanFilter_and_geq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_and_leq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == EQ_VEC)
                            genScanFilter_and_eq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == NOT_EQ_VEC)
                            genScanFilter_and_neq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GTH_VEC)
                            genScanFilter_and_gth_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LTH_VEC)
                            genScanFilter_and_lth_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GEQ_VEC)
                            genScanFilter_and_geq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LEQ_VEC)
                            genScanFilter_and_leq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == IN)
                            genScanFilter_and_in<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == IN_VEC)
                            genScanFilter_and_in_vec<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LIKE)
                            genScanFilter_and_like<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
                    }else{
                        if (rel == EQ)
                            genScanFilter_or_eq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == NOT_EQ)
                            genScanFilter_or_neq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GTH)
                            genScanFilter_or_gth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LTH)
                            genScanFilter_or_lth<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GEQ)
                            genScanFilter_or_geq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LEQ)
                            genScanFilter_or_leq<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == EQ_VEC)
                            genScanFilter_or_eq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == NOT_EQ_VEC)
                            genScanFilter_or_neq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GTH_VEC)
                            genScanFilter_or_gth_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LTH_VEC)
                            genScanFilter_or_lth_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == GEQ_VEC)
                            genScanFilter_or_geq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LEQ_VEC)
                            genScanFilter_or_leq_vec<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == IN)
                            genScanFilter_or_in<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == IN_VEC)
                            genScanFilter_or_in_vec<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
                        else if (rel == LIKE)
                            genScanFilter_or_like<<<grid, block>>>(column[whereIndex], sn->tn->attrSize[index], totalTupleNum, gpuExp, gpuFilter);
                    }
                }

            }else if(format == DICT){

                struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
                dNum = dheader->dictNum;
                byteNum = dheader->bitNum/8;

                struct dictHeader * gpuDictHeader;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

                if(dictFilter != -1){
                    if(where->andOr == AND)
                        genScanFilter_dict_and<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
                    else
                        genScanFilter_dict_or<<<grid,block>>>(gpuDictHeader,sn->tn->attrSize[index],sn->tn->attrType[index],dNum, gpuExp,gpuDictFilter);
                }
                dictFilter = 0;

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

            }else if (format == RLE){
                genScanFilter_rle<<<grid,block>>>(column[whereIndex],sn->tn->attrSize[index],sn->tn->attrType[index], totalTupleNum, gpuExp, where->andOr, gpuFilter);

            }

        }

        //Stop timer for Step 1 - Copy where clause (Part2)
        CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
        clock_gettime(CLOCK_REALTIME, &endS12);
        pp->whereMemCopy_s1 += (endS12.tv_sec - startS12.tv_sec)* BILLION + endS12.tv_nsec - startS12.tv_nsec;            

        if(prevFormat == DICT){
            if (dictInit == 1){
                transform_dict_filter_init<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
                dictInit = 0;
            }else if(dictFinal == AND)
                transform_dict_filter_and<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
            else
                transform_dict_filter_or<<<grid,block>>>(gpuDictFilter, column[prevWhere], totalTupleNum, dNum, gpuFilter, byteNum);
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
        }

        if(whereFree[prevWhere] == 1 && (sn->tn->dataPos[prevIndex] == MEM || sn->tn->dataPos[prevIndex] == MMAP || sn->tn->dataPos[prevIndex] == PINNED))
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(column[prevWhere]));

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuExp));

    }

    //Start timer for Step 4 - Count result (PreScan)
    struct timespec startS4, endS4;
    clock_gettime(CLOCK_REALTIME,&startS4);
 

    /* --Optimization Note--
    * We cannot remove this one even if we know the number of selected tuples
    * because counts the tuples per thread (not only total number)
    */

    /* 
    * Count the number of tuples that meets the predicats for each thread
    * and calculate the prefix sum.
    */
    countScanNum<<<grid,block>>>(gpuFilter,totalTupleNum,gpuCount);
    scanImpl(gpuCount,threadNum, gpuPsum, pp);

    //Stop timer for Step 4 - Count result (PreScan)
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS4);
    pp->preScanTotal_s4 += (endS4.tv_sec -  startS4.tv_sec)* BILLION + endS4.tv_nsec - startS4.tv_nsec;
    pp->preScanCount_s4++;

    int tmp1, tmp2;

    //Start timer for Step 5 - Count result (PreScan)
    struct timespec startS5, endS5;
    clock_gettime(CLOCK_REALTIME,&startS5);

    if (idxPos < 0){ 
    
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp1, &gpuCount[threadNum-1], sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp2, &gpuPsum[threadNum-1], sizeof(int), cudaMemcpyDeviceToHost));
        
        count = tmp1+tmp2;
        res->tupleNum = count;
    }

    //End timer for Step 5 - Count result (PreScan)
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS5);
    pp->preScanResultMemCopy_s5 += (endS5.tv_sec -  startS5.tv_sec)* BILLION + endS5.tv_nsec - startS5.tv_nsec;
    
    //Start timer for Other - 02 (mallocRes)
    struct timespec startS02,endS02;
    clock_gettime(CLOCK_REALTIME,&startS02);
    
    //CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuCount));

    char **result = NULL, **scanCol = NULL;

    attrNum = sn->outputNum;

    scanCol = (char **) malloc(attrNum * sizeof(char *));
    CHECK_POINTER(scanCol);
    result = (char **) malloc(attrNum * sizeof(char *));
    CHECK_POINTER(result);

    //Stop timer for Other - 02 (mallocRes)
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS02);
    pp->mallocRes_S02 += (endS02.tv_sec - startS02.tv_sec)* BILLION + endS02.tv_nsec - startS02.tv_nsec;

    //Start timer for Step 6 - Copy to device all other columns
    struct timespec startS6, endS6;
    clock_gettime(CLOCK_REALTIME,&startS6);

    for(int i=0;i<attrNum;i++){

        int pos = colWherePos[i];
        int index = sn->outputIndex[i];
        tupleSize += sn->tn->attrSize[index];

        if(pos != -1){
            scanCol[i] = column[pos];
        }else{
            if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &scanCol[i] , sn->tn->attrTotalSize[index]));

            if(sn->tn->dataFormat[index] != DICT){
                if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(scanCol[i], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
                else
                    scanCol[i] = sn->tn->content[index];

            }else{
                if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
                    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(scanCol[i], sn->tn->content[index], sn->tn->attrTotalSize[index], cudaMemcpyHostToDevice));
                else
                    scanCol[i] = sn->tn->content[index];
            }
        }

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &result[i], count * sn->tn->attrSize[index]));
    }

    //End timer for Step 6 - Copy to device all other columns
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS6);
    pp->dataMemCopyOther_s6 += (endS6.tv_sec -  startS6.tv_sec)* BILLION + endS6.tv_nsec - startS6.tv_nsec;
   
    //Start timer for Step 7 - Materialize result
    struct timespec startS7, endS7;
    clock_gettime(CLOCK_REALTIME,&startS7);

    if(1){

        for(int i=0; i<attrNum; i++){
            int index = sn->outputIndex[i];
            int format = sn->tn->dataFormat[index];
            if(format == UNCOMPRESSED){
                if (sn->tn->attrSize[index] == sizeof(int))
                    scan_int<<<grid,block>>>(scanCol[i], sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);
                else
                    scan_other<<<grid,block>>>(scanCol[i], sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

            }else if(format == DICT){
                struct dictHeader * dheader = (struct dictHeader *)sn->tn->content[index];
                int byteNum = dheader->bitNum/8;

                struct dictHeader * gpuDictHeader = NULL;
                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader,sizeof(struct dictHeader)));
                CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader,dheader,sizeof(struct dictHeader), cudaMemcpyHostToDevice));

                if (sn->tn->attrSize[i] == sizeof(int))
                    scan_dict_int<<<grid,block>>>(scanCol[i], gpuDictHeader, byteNum,sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);
                else
                    scan_dict_other<<<grid,block>>>(scanCol[i], gpuDictHeader,byteNum,sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

            }else if(format == RLE){
                int dNum = (sn->tn->attrTotalSize[index] - sizeof(struct rleHeader))/(3*sizeof(int));
                char * gpuRle = NULL;

                CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRle, totalTupleNum * sizeof(int)));

                unpack_rle<<<grid,block>>>(scanCol[i], gpuRle,totalTupleNum, dNum);


                scan_int<<<grid,block>>>(gpuRle, sn->tn->attrSize[index], totalTupleNum,gpuPsum,count, gpuFilter,result[i]);

                CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));
            }

        }
    }

    //End timer for Step 7 - Materialize result
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS7);
    pp->materializeResult_s7 += (endS7.tv_sec -  startS7.tv_sec)* BILLION + endS7.tv_nsec - startS7.tv_nsec;
   
        
    //Start timer for Step 8 - Copy final result
    struct timespec startS8, endS8;
    clock_gettime(CLOCK_REALTIME,&startS8);

    res->tupleSize = tupleSize;

    for(int i=0;i<attrNum;i++){

        int index = sn->outputIndex[i];

        if(sn->tn->dataPos[index] == MEM || sn->tn->dataPos[index] == MMAP || sn->tn->dataPos[index] == PINNED)
            CUDA_SAFE_CALL_NO_SYNC(cudaFree(scanCol[i]));

        int colSize = res->tupleNum * res->attrSize[i];

        res->attrTotalSize[i] = colSize;
        res->dataFormat[i] = UNCOMPRESSED;

        if(sn->keepInGpu == 1){
            res->dataPos[i] = GPU;
            res->content[i] = result[i];
        }else{
            res->dataPos[i] = MEM;
            res->content[i] = (char *)malloc(colSize);
            CHECK_POINTER(res->content[i]);
            memset(res->content[i],0,colSize);
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i],result[i],colSize ,cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaFree(result[i]));
        }
    }

    //End timer for Step 8 - Copy final result
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS8);
    pp->finalResultMemCopy_s8 += (endS8.tv_sec -  startS8.tv_sec)* BILLION + endS8.tv_nsec - startS8.tv_nsec;
   
    //Start timer for Other - 03 (deallocate buffers)
    struct timespec startS03,endS03;
    clock_gettime(CLOCK_REALTIME,&startS03);

    //CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuPsum));
    //CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuFilter));

    //free(column);
    free(scanCol);
    free(result);

    //Stop timer for Other - 01 (tableScan)
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME, &endS03);
    pp->deallocateBuffs_S03 += (endS03.tv_sec - startS03.tv_sec)* BILLION + endS03.tv_nsec - startS03.tv_nsec;

    //Stop timer (Global tableScan())
    CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
    clock_gettime(CLOCK_REALTIME,&endTableScanTotal);
    pp->tableScanTotal += (endTableScanTotal.tv_sec -  startTableScanTotal.tv_sec)* BILLION + endTableScanTotal.tv_nsec - startTableScanTotal.tv_nsec;
    pp->tableScanCount ++;

    return res;
}