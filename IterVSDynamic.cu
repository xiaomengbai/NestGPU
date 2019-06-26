/*
query from w3schools
SELECT OrderDate FROM Orders o
WHERE o.CustomerID IN (
  SELECT CustomerID FROM Customers c
  WHERE o.OrderID IN (
    SELECT OrderID
    FROM OrderDetails od, Products p,Employees e
    WHERE od.ProductID = p.ProductID
    AND o.EmployeeID = e.EmployeeID));

query used in this file which is similar with the one above

SELECT a.attr FROM a
WHERE a.ckey IN (
  SELECT c.key FROM c
  WHERE a.bkey IN (
    SELECT b.key FROM b
    WHERE a.attr = b.attr));
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <algorithm>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

using std::string;

#define MAX_INT 1024*1024*1024
#define BILLION             1000000000

#define CUDACHECK(cmd) do { \
    cudaError_t e = cmd; \
    pid_t pid = getpid();\
    if( e != cudaSuccess ) { \
    printf("Porcess %d Failed: Cuda error %s:%d '%s'\n", \
    pid,__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
    } \
} while(0)

#define NP2(n)              do {                    \
n--;                                            \
n |= n >> 1;                                    \
n |= n >> 2;                                    \
n |= n >> 4;                                    \
n |= n >> 8;                                    \
n |= n >> 16;                                   \
n ++; } while (0)

#define TABLE2 0x8000000000000000
template<typename vec_t>
__global__ static void find_MSB(vec_t* data,int * msb,int size)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;
    vec_t tmp,t_data;
    int l_msb = 0;

    __shared__ int s_msb;
    __shared__ unsigned	long long max_data;

    s_msb = 0;
    max_data = 0;

    __syncthreads();

    for (int i = offset; i<size; i += stride){
    	t_data = data[offset];
    	if(max_data < t_data)
    	{
	    	for(int s = 0; s < 64; s++)
	    	{
		    	tmp = t_data >> s;
		        if(tmp != 0) l_msb = s;
		        else break;
		    }
		    atomicMax(&s_msb,l_msb);
		    atomicMax(&max_data,t_data);
		}
    }

    __syncthreads();

    if(threadIdx.x == 0)
    	atomicMax(&*msb,s_msb);
}

template<typename vec_t>
__global__ static void find_MAX(vec_t* data,vec_t * msb,int size)
{
    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;
    vec_t t_data;

    __shared__ vec_t max_data;

    max_data = 0;

    __syncthreads();

    for (int i = offset; i<size; i += stride){
        t_data = data[offset];
        if(max_data < t_data)
            atomicMax(&max_data,t_data);
    }

    __syncthreads();

    if(threadIdx.x == 0)
        atomicMax(&*msb,max_data);
}

template<typename vec_t>
__global__ static void assign_index(vec_t *dim, long  inNum){
    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = offset; i<inNum; i += stride)
        dim[i] = i;
}

__global__ static void count_hash_num(int *dim, long  inNum,int *num,int hsize){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *)dim)[i];
        int hKey = joinKey & (hsize-1);
        atomicAdd(&(num[hKey]),1);
    }
}

__global__ static void see(int * data)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;
}

__global__ static void build_hash_table(int *dim, long inNum, int *psum, int * bucket,int hsize){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *) dim)[i];
        int hKey = joinKey & (hsize-1);
        int pos = atomicAdd(&psum[hKey],1) * 2;
        assert(pos < inNum * 2);
        ((int*)bucket)[pos] = joinKey;
        pos += 1;
        int dimId = i+1;
        ((int*)bucket)[pos] = dimId;
    }

}


__global__ static void count_join_result(int* num, int* psum, int* bucket, int* fact, long inNum, int* count, int * factFilter,int hsize){

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

            if(dimKey == fkey){

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

__global__ static void materialization(int* data, int* psum, long inNum, int * factFilter, int * result)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;
    int localCount = psum[offset];

    for(int i=offset;i<inNum;i+=stride){
    	int dimID = factFilter[i];
    	if(dimID != 0)
    	{
    		((int*)result)[localCount] = ((int *)data)[i];
            localCount ++;
    	}
    }
}

__global__ static void right_materialization(int* data, int* psum, long inNum, int * factFilter, int * result)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;
    int localCount = psum[offset];

    for(int i=offset;i<inNum;i+=stride){
    	int dimID = factFilter[i];
    	if(dimID != 0)
    	{
    		((int*)result)[localCount] = ((int *)data)[dimID];
            localCount ++;
    	}
    }
}

__global__ static void scanCol(int inNum, int * result1, int * result2, int * filter)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
    	if(result1[i] == result2[i])
    	{
    		filter[i] = 1;
    	}
    }
}

__global__ static void scanCol_withIndex(int inNum, int * result1, int * result2, int r_num, int * filter, int * final_filter)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=0;i<inNum;i++){
    	if(filter[i] == 1)
    	{
    		for(int t = offset; t < r_num; t += stride)
    		{
    			if(result1[i] == result2[t])
    				final_filter[t] = 1;
    		}
    	}
    }
}


__global__ static void build_hash_table_dual(int *dim, int * dim_index, long inNum, int *psum, int * bucket,int hsize){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *) dim)[i];
        int joinAttr = ((int *) dim_index)[i];
        int hKey = joinKey & (hsize-1);
        int pos = atomicAdd(&psum[hKey],1) * 3;
        assert(pos < inNum * 3);
        ((int*)bucket)[pos] = joinKey;
        pos += 1;
        ((int*)bucket)[pos] = joinAttr;
        pos += 1;
        int dimId = i+1;
        ((int*)bucket)[pos] = dimId;
    }
}

__global__ static void count_join_result_dual(int* num, int* psum, int* bucket, int* fact, int * fact_attr, long inNum, bool* count,int hsize){

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
            int dimKey = ((int *)(bucket))[3*j + 3*pSum];

            if(dimKey == fkey){
                int dimAttr = ((int *)(bucket))[3*j + 3*pSum + 1];
                int factAttr = ((int *)(fact_attr))[i];
                if(dimAttr == factAttr)
                {
                	count[i] = true;
                	break;
            	}
            }
        }
    }
}


__global__ static void compare(int * a, int *b, int size)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<size;i+=stride){
    	if(a[i] != b[i])
    	{
    		printf("error");
    		assert(a[i] == b[i]);
    	}
    }
}


__global__ static void recallScanUmlimited(
     int * allCol, int outerIndex, int currentLevel, // aggFunc: -1 is no agg, 0 is MAX/MIN, 1 is SUM, 2 is AVG
     int loopLevel, int * attrSize, int * attrNum, int * attrIdx, int * psum, int * attrOP, bool * innerOuterMatchingBitmap, int * interRE) //Output
{
	/*
    stuff used in this kernel
    //the numbers refer to the colomns and the MAX_INT means this is a subquery
    int attrNeeded[] = {2,1,MAX_INT,5,0,MAX_INT,3,2,4};
    int attrPsum[] = {0,3,6};

    //0 for project this attr, 1 for left of IN, 2 for right IN, 3 for left of join, 4 for right of join
    //same length with the attrNeeded, one to one correspondence
    int attrOp[]   = {0,1,2,0,1,2,0,3,4};
    */

	int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int proIdx, leftIdx, rightIdx;
    int leftValue, rightValue, proValue;
    int valueBegin, tableIndex;
    int innerBegin, innerIndex;

    int loopIndex;

    if(currentLevel < loopLevel)
    {
	    int planStart = psum[currentLevel];
	    int planEnd   = (currentLevel == loopLevel - 1 ? psum[currentLevel + 1] : 9);

	    for(int i = planStart; i < planEnd; i++)
	    {
	    	if(attrOP[i] == 0)
	    		proIdx = attrIdx[i];
	    	else if(attrOP[i] == 1 || attrOP[i] == 3)
	    		leftIdx = attrIdx[i];
	    	else if(attrOP[i] == 2)
	    	{
	    		for(int t = tid; t < attrSize[0]; t += stride)
	    		{
	    			loopIndex = currentLevel == 0? t : outerIndex;
	    			recallScanUmlimited<<<32,128>>>(allCol, t, currentLevel + 1, loopLevel, attrSize, attrNum,attrIdx, psum, attrOP,innerOuterMatchingBitmap,interRE);
	    			//after the recursive, scan interRE
	    			leftValue = allCol[leftIdx * attrSize[0] + loopIndex];
	    			if(innerOuterMatchingBitmap[t] == true || leftValue == interRE[t])
	    			{
	    				innerOuterMatchingBitmap[loopIndex] = true;
	    				if(proIdx >= 3)
					    {
					    	proIdx -= 3;
					    	tableIndex = 2;
					    	valueBegin = attrNum[0] * attrSize[0];
					    }
					    else
					    {
					    	tableIndex = 0;
					    	valueBegin = 0;
					    }
	    				proValue = allCol[valueBegin + attrSize[1] * proIdx + t];
	    				interRE[loopIndex] = proValue;
	    				return;
	    			}
	    		}
	    		if(currentLevel != 0)
	    			innerOuterMatchingBitmap[outerIndex] = false;
	    	}
	    	else if(attrOP[i] == 4)//join processing
	    	{
	    		rightIdx = attrIdx[i];

	    		if(rightIdx >= 3)
			    {
			    	rightIdx -= 3;
			    	tableIndex = 2;
			    	valueBegin = attrNum[0] * attrSize[0];
			    }
			    else
			    {
			    	tableIndex = 0;
			    	valueBegin = 0;
			    }

	    		for(int t = tid; t < attrSize[1] ; t += stride)
	    		{
	    			loopIndex = currentLevel == 0? t : outerIndex;
	    			leftValue = allCol[leftIdx * attrSize[0] + loopIndex];
	    			rightValue = allCol[valueBegin + attrSize[1] * rightIdx + t];
	    			if(leftValue == rightValue)
	    			{
	    				//store the corresponding results in the innerOuterMatchingBitmap and interRE
	    				innerOuterMatchingBitmap[loopIndex] = true;
	    				if(proIdx >= 3)
					    {
					    	proIdx -= 3;
					    	valueBegin = attrNum[0] * attrSize[0];
					    }
					    else
					    {
					    	// tableIndex = 0;
					    	valueBegin = 0;
					    }
	    				proValue = allCol[valueBegin + attrSize[1] * proIdx + t];
	    				interRE[loopIndex] = proValue;
	    				return;
	    			}
	    		}
	    	}
	    }
	}
}

cudaError_t Subquery_proc(int *a_bkey,int * a_ckey,int * a_attr,int lsize,int * b_key,int * b_attr,int rsize, int * c_key, int c_size)
{
    struct timespec start_t, end_t;
    int defaultBlock = 4096;
    double total = 0.0;
    double timeE = 0.0;
    cudaStream_t s;
    CUDACHECK(cudaStreamCreate(&s));

    dim3 grid(defaultBlock);
    dim3 block(256);
    int myRank = 0;
    int hsize = rsize;
    NP2(hsize);

    int *gpu_dim_psum = NULL, *gpu_fact_psum = NULL, *gpu_dim_hashNum = NULL;
    int total_count = 0;
    int all_count = 0;
    int * d_max, max, mask,*gpu_count_db = NULL,*gpu_count_new = NULL;
    // int alloc_size = rsize > lsize ? rsize : lsize;
    int * factFilter_db = NULL, * bucket = NULL, * filter = NULL;
    int * filterResult1 = NULL, * filterResult2 = NULL, * filterResult3 = NULL, * gpu_resPsum = NULL, * rightResult = NULL;
    bool * mapbit =NULL;
    clock_gettime(CLOCK_REALTIME,&start_t);

    CUDACHECK(cudaMalloc((void **)&gpu_dim_hashNum, hsize * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&gpu_dim_psum, hsize * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&gpu_fact_psum, hsize * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&d_max, sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&gpu_count_db, 4096*256*sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&gpu_resPsum, 4096*256*sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&factFilter_db, lsize * sizeof(int)));
    CUDACHECK(cudaMemset(gpu_dim_psum,0, hsize * sizeof(int)));
    CUDACHECK(cudaMemset(gpu_fact_psum,0, hsize * sizeof(int)));

    CUDACHECK(cudaMalloc((void **)&mapbit, lsize * sizeof(bool)));

    CUDACHECK(cudaMalloc((void **)&bucket, rsize * 2 * sizeof(int)));

    cudaMemset(mapbit, 0, lsize * sizeof(bool));

    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("init Time: %lf ms hsize %d full block size %d\n", timeE/(1000*1000),hsize,grid.x);

    //three table size
    int attrSize[3] = {lsize, rsize, rsize};
    //number of attrs in each table
    int attrNum[3] = {3, 2, 1};
    // only two loops now
    int loopLevel = 3;
    string ** attrIdx = NULL, ** opera = NULL;
    attrIdx = (string **)malloc(sizeof(string *) * loopLevel);
    opera   = (string **)malloc(sizeof(string *) * loopLevel);
    for(int i = 0; i < loopLevel; i ++)
    {
    	attrIdx[i] = (string *)malloc(sizeof(string) * 10);
    	opera[i]   = (string *)malloc(sizeof(string) * 10);
    }

    int attrTotalSize = attrNum[0] * attrSize[0] + attrNum[1] * attrSize[1] + attrNum[2] * attrSize[2];
    //some thing like the plan
    attrIdx[0][0] = "a_attr";
    attrIdx[0][1] = "a_ckey";
    attrIdx[0][2] = "sub";
    attrIdx[0][3] = "end";

    attrIdx[1][0] = "c_key";
    attrIdx[1][1] = "a_bkey";
    attrIdx[1][2] = "sub";
    attrIdx[1][3] = "end";

    attrIdx[1][0] = "b_key";
    attrIdx[1][1] = "a_attr";
    attrIdx[1][2] = "b_attr";
    attrIdx[1][3] = "end";

    opera[0][1] = "pro"; opera[0][1] = "in", opera[0][1] = "1"; opera[0][1] = "end";
    opera[1][1] = "pro"; opera[1][1] = "in", opera[1][1] = "2"; opera[1][1] = "end";
    opera[2][1] = "pro"; opera[1][1] = "join", opera[1][1] = "-1"; opera[1][1] = "end";

    cudaMemset(mapbit, 0, lsize * sizeof(bool));

    int currentLevel = 0,currentOP = 0;
    int * leftTable = NULL, * rightTable = NULL;
    //GPU-DB like processing, hard coding, bottom up
    clock_gettime(CLOCK_REALTIME,&start_t);
    CUDACHECK(cudaMemset(gpu_dim_hashNum,0, hsize * sizeof(int)));
    CUDACHECK(cudaMemset(factFilter_db,0, lsize * sizeof(int)));
    CUDACHECK(cudaMemset(gpu_count_db,0, 4096*256 * sizeof(int)));

    count_hash_num<<<4096,256>>>(b_attr,rsize,gpu_dim_hashNum,hsize);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::exclusive_scan(thrust::device, gpu_dim_hashNum, gpu_dim_hashNum + hsize, gpu_dim_psum); // in-place scan

    cudaMemcpy(gpu_fact_psum, gpu_dim_psum, hsize * sizeof(int), cudaMemcpyDeviceToDevice);

    build_hash_table<<<4096,256>>>(b_attr,rsize,gpu_fact_psum,bucket,hsize);

    CUDACHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("build Time: %lf ms \n",  timeE/(1000*1000));
    total += timeE;

    clock_gettime(CLOCK_REALTIME,&start_t);

    count_join_result<<<4096,256>>>(gpu_dim_hashNum,gpu_dim_psum,bucket,a_attr,lsize,gpu_count_db,factFilter_db,hsize);
    int tmp1, tmp2;
    CUDACHECK(cudaDeviceSynchronize());
    see<<<1,1>>>(gpu_count_db);
    CUDACHECK(cudaMemcpy(&tmp1,&gpu_count_db[4096*256-1],sizeof(int),cudaMemcpyDeviceToHost));
    thrust::exclusive_scan(thrust::device, gpu_count_db, gpu_count_db + 4096*256, gpu_resPsum);
    CUDACHECK(cudaMemcpy(&tmp2,&gpu_resPsum[4096*256-1],sizeof(int),cudaMemcpyDeviceToHost));

    int resCount = tmp1 + tmp2;

    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("%d rows probe Time: %lf ms \n", resCount, timeE/(1000*1000));
    total += timeE;

    clock_gettime(CLOCK_REALTIME,&start_t);

    CUDACHECK(cudaMalloc((void **)&filterResult1, resCount * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&filterResult2, resCount * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&filterResult3, resCount * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&rightResult, resCount * sizeof(int)));

    materialization<<<4096,256>>>(a_bkey, gpu_resPsum, lsize, factFilter_db, filterResult1);
    materialization<<<4096,256>>>(a_ckey, gpu_resPsum, lsize, factFilter_db, filterResult2);
    materialization<<<4096,256>>>(a_attr, gpu_resPsum, lsize, factFilter_db, filterResult3);
    right_materialization<<<4096,256>>>(b_key,  gpu_resPsum, lsize, factFilter_db, rightResult);

    int * a_bkey_new, * a_ckey_new, * a_attr_new;
    a_bkey_new = filterResult1;
    a_ckey_new = filterResult2;
    a_attr_new = filterResult3;

    cudaFree(factFilter_db);

    CUDACHECK(cudaMalloc((void **)&factFilter_db, resCount * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&filter, resCount * sizeof(int)));
    CUDACHECK(cudaMemset(factFilter_db,0, resCount * sizeof(int)));
    CUDACHECK(cudaMemset(filter,0, resCount * sizeof(int)));

    CUDACHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("materialization Time: %lf ms \n",  timeE/(1000*1000));
    total += timeE;

    clock_gettime(CLOCK_REALTIME,&start_t);
    //IN predicate processing
    scanCol<<<4096,256>>>(resCount, a_bkey_new, rightResult,factFilter_db);

    scanCol_withIndex<<<4096,256>>>(resCount, a_ckey_new, c_key, c_size, factFilter_db, filter);

    CUDACHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("scan Time: %lf ms \n", timeE/(1000*1000));
    total += timeE;

    cudaFree(filterResult1);
    cudaFree(filterResult2);
    cudaFree(filterResult3);

    printf("GPU-DB like nest Time: %lf ms\n\n", total/(1000*1000));

    //dynamic parallelism, I think it is right, but it just can not stop.
    clock_gettime(CLOCK_REALTIME,&start_t);

    int * allAttr = NULL, * roundAttr = NULL, * d_attrSize = NULL, * d_attrNum = NULL;
    int * d_attrNeeded = NULL, * d_attrOp = NULL, * d_psum = NULL, * intermediaRE = NULL;

    int attrNeeded[] = {2,1,MAX_INT,5,0,MAX_INT,3,2,4};
    int attrPsum[] = {0,3,6};
    //0 for project, 1 for left of IN, 2 for right IN, 3 for left of join, 4 for right of join
    int attrOp[]   = {0,1,2,0,1,2,0,3,4};
    CUDACHECK(cudaMalloc((void **)&allAttr, attrTotalSize * sizeof(int)));
    // CUDACHECK(cudaMalloc((void **)&roundAttr, 3 * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&d_attrSize, 3 * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&d_attrNum, 3 * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&d_attrNeeded, 9 * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&d_attrOp, 9 * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&d_psum, 3 * sizeof(int)));
    CUDACHECK(cudaMalloc((void **)&intermediaRE, lsize * sizeof(int)));

    CUDACHECK(cudaMemcpy(d_attrSize,attrSize, 3 * sizeof(int),cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_attrNum,attrNum, 3 * sizeof(int),cudaMemcpyHostToDevice));

    cudaMemcpy(d_attrNeeded,attrNeeded, 9 * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_attrOp,attrOp, 9 * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_psum,attrPsum, 3 * sizeof(int),cudaMemcpyHostToDevice);

    attrTotalSize = 0;
    int aggFunc = 0;

    cudaMemcpy(allAttr,a_bkey, lsize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(allAttr + lsize,a_ckey, lsize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(allAttr + 2 * lsize,a_attr, lsize * sizeof(int), cudaMemcpyDeviceToDevice);

    attrTotalSize += attrNum[0] * attrSize[0];

    cudaMemcpy(allAttr + attrTotalSize,b_key, rsize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(allAttr + attrTotalSize + rsize,b_attr, rsize * sizeof(int), cudaMemcpyDeviceToDevice);

    attrTotalSize += attrNum[1] * attrSize[1];

    cudaMemcpy(allAttr + attrTotalSize,c_key, c_size * sizeof(int), cudaMemcpyDeviceToDevice);

    clock_gettime(CLOCK_REALTIME,&start_t);

    cudaMemset(mapbit, 0, lsize * sizeof(bool));
    cudaMemset(intermediaRE, 0, lsize * sizeof(int));

    int roundLevel = 3;

    recallScanUmlimited<<<256,128>>>(allAttr, 0, 0, roundLevel, d_attrSize,d_attrNum,d_attrNeeded,d_psum, d_attrOp, mapbit, intermediaRE);

    // nestScanDynamicUnlimited_deviceFunc<<<32,128>>>(allAttr,roundAttr,roundLevel,d_attrSize,d_attrNum,d_attrNeeded,d_psum, d_attrOp, 0,mapbit, intermediaRE);

    CUDACHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("dynamic scan Time: %lf ms \n", timeE/(1000*1000));
    total += timeE;

    cudaError_t cudaStatus = cudaDeviceSynchronize();

    if(cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Subquery error\n");
        CUDACHECK(cudaStatus);
    }


    total += timeE;

    clock_gettime(CLOCK_REALTIME, &start_t);
    // CUDACHECK(cudaFree(gpu_data));
    // CUDACHECK(cudaFree(gpu_index));
    CUDACHECK(cudaFree(gpu_dim_hashNum));
    CUDACHECK(cudaFree(gpu_dim_psum));
    CUDACHECK(cudaFree(gpu_fact_psum));
    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;

    printf("local mem free Time: %lf ms\n", timeE/(1000*1000));
    total += timeE;

    cudaStatus = cudaDeviceSynchronize();
    CUDACHECK(cudaStatus);
    return cudaStatus;
}


cudaError_t Preparation(int *a_bkey, int * a_ckey, int * a_attr, int l_size, int * b_key, int * b_attr, int r_size, int * c_key, int c_size, int argc, char* argv[]);

int main(int argc, char* argv[])
{
	int a_size = 4*1024*1024;
	int b_size = 1*1024*1024;

	if(argc >= 2)
	{
		if(atoi(argv[1]) != 0)
		{
			a_size = atoi(argv[1])*1024*1024;
		}
	}

	if(argc >= 3)
	{
		if(atoi(argv[2])!=0)
			b_size = atoi(argv[2])*1024*1024;
	}
	int *a_bkey,*a_ckey,*a_attr;
	int *b_key,*b_attr;
	int *c_key;
	int c_size = b_size;

	fprintf(stderr,"R relation size %d Rows , S relation size %d \n",a_size,b_size);

	CUDACHECK(cudaHostAlloc((void **)&a_bkey, sizeof(int)*a_size, cudaHostAllocPortable | cudaHostAllocMapped));
	CUDACHECK(cudaHostAlloc((void **)&b_key, sizeof(int)*b_size, cudaHostAllocPortable | cudaHostAllocMapped));
	CUDACHECK(cudaHostAlloc((void **)&a_ckey, sizeof(int)*a_size, cudaHostAllocPortable | cudaHostAllocMapped));
	CUDACHECK(cudaHostAlloc((void **)&b_attr, sizeof(int)*b_size, cudaHostAllocPortable | cudaHostAllocMapped));
	CUDACHECK(cudaHostAlloc((void **)&a_attr, sizeof(int)*a_size, cudaHostAllocPortable | cudaHostAllocMapped));
	CUDACHECK(cudaHostAlloc((void **)&c_key, sizeof(int)*c_size, cudaHostAllocPortable | cudaHostAllocMapped));

	for (int i = 0; i < b_size; i++)
	{
		b_key[i] = i;
		b_attr[i] = rand();
	}

	for (int i = 0; i < c_size; i++)
	{
		c_key[i] = i + 1000000;
	}

	for (int i = 0; i < a_size; i++)
	{
		int tmp = rand()%b_size;
		if(!tmp)
			tmp ++;
		a_bkey[i] = tmp;
		a_ckey[i] = tmp + 1000000;
		a_attr[i] = rand();
	}
	cudaError_t cudaStatus;

	cudaStatus = Preparation(a_bkey, a_ckey, a_attr, a_size,b_key, b_attr, b_size, c_key, c_size, argc, argv);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Subquery failed!\n");
		return 1;
	}

	return 0;
}

template<typename vec_t>
__global__ static void assign_index(vec_t *dim, long  inNum,int rank){
    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = offset; i<inNum; i += stride)
        dim[i] = i;
}

cudaError_t Preparation(int *a_bkey, int * a_ckey, int * a_attr, int l_size, int * b_key, int * b_attr, int r_size, int * c_key, int c_size, int argc, char* argv[])
{
	struct timespec start, end;
	struct timespec start_t, end_t;
	cudaError_t cudaStatus;
	cudaEvent_t event;
	cudaIpcMemHandle_t memHandle_t;

	int defaultBlock = 4096;
	int new_size = 0;
	dim3 grid(defaultBlock);
	dim3 block(256);
	cudaStream_t s;
	CUDACHECK(cudaStreamCreate(&s));

	double total = 0.0;

	clock_gettime(CLOCK_REALTIME,&start);
	int * gpu_a_bkey = NULL, * gpu_a_attr = NULL, * gpu_a_ckey = NULL;
	int * gpu_b_key  = NULL, * gpu_b_attr = NULL, * gpu_c_key  = NULL;

	clock_gettime(CLOCK_REALTIME,&start_t);
	int primaryKeySize = l_size * sizeof(int);
	int filterSize = r_size * sizeof(int);

	CUDACHECK(cudaMalloc((void **)&gpu_a_bkey, primaryKeySize));
	CUDACHECK(cudaMalloc((void **)&gpu_a_ckey, primaryKeySize));
	CUDACHECK(cudaMalloc((void **)&gpu_a_attr, primaryKeySize));
	CUDACHECK(cudaMalloc((void **)&gpu_b_key, filterSize));
	CUDACHECK(cudaMalloc((void **)&gpu_b_attr, filterSize));
	CUDACHECK(cudaMalloc((void **)&gpu_c_key, filterSize));

	clock_gettime(CLOCK_REALTIME,&end_t);
	double timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
	total += timeE;

	printf("init Time: %lf ms\n", timeE/(1000*1000));

	clock_gettime(CLOCK_REALTIME,&start_t);
	CUDACHECK(cudaMemcpyAsync(gpu_a_bkey, a_bkey, primaryKeySize, cudaMemcpyHostToDevice, s));
	CUDACHECK(cudaMemcpyAsync(gpu_a_ckey, a_ckey, primaryKeySize, cudaMemcpyHostToDevice, s));
	CUDACHECK(cudaMemcpyAsync(gpu_a_attr, a_attr, primaryKeySize, cudaMemcpyHostToDevice, s));
	CUDACHECK(cudaMemcpyAsync(gpu_b_key,  b_key , filterSize, cudaMemcpyHostToDevice, s));
	CUDACHECK(cudaMemcpyAsync(gpu_b_attr, b_attr, filterSize, cudaMemcpyHostToDevice, s));
	CUDACHECK(cudaMemcpyAsync(gpu_c_key,  c_key , filterSize, cudaMemcpyHostToDevice, s));

	CUDACHECK(cudaDeviceSynchronize());

	clock_gettime(CLOCK_REALTIME,&end_t);
	timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
	total += timeE;

	printf("Host To Device Time: %lf ms\n", timeE/(1000*1000));

	clock_gettime(CLOCK_REALTIME, &start_t);

	cudaStatus = Subquery_proc(gpu_a_bkey,gpu_a_ckey,gpu_a_attr,l_size,gpu_b_key,gpu_b_attr,r_size,gpu_c_key,c_size);

	clock_gettime(CLOCK_REALTIME,&end_t);
	timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
	total += timeE;

	printf("Subquery Time: %lf ms\n", timeE/(1000*1000));

	clock_gettime(CLOCK_REALTIME, &start_t);
	CUDACHECK(cudaFree(gpu_a_bkey));
	CUDACHECK(cudaFree(gpu_a_ckey));
	CUDACHECK(cudaFree(gpu_a_attr));
	CUDACHECK(cudaFree(gpu_b_key));
	CUDACHECK(cudaFree(gpu_b_attr));
	CUDACHECK(cudaFree(gpu_c_key));

	clock_gettime(CLOCK_REALTIME,&end_t);
	timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
	printf("second GPU original memory free Time: %lf ms\n", timeE/(1000*1000));
	total += timeE;

	clock_gettime(CLOCK_REALTIME, &start_t);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME,&end_t);
	timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
	printf("second CPU memory free Time: %lf ms\n", timeE/(1000*1000));
	total += timeE;

	clock_gettime(CLOCK_REALTIME,&end);
	timeE = (end.tv_sec -  start.tv_sec)* BILLION + end.tv_nsec - start.tv_nsec;
	printf("Whole Processing Time: %lf ms Whole time : %1f ms \n", total/(1000*1000),timeE/(1000*1000));

	return cudaDeviceSynchronize();
}