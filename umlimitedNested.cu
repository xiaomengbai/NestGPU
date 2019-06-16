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
#include <cub/cub.cuh>
#include <cub/grid/grid_barrier.cuh>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

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

template<typename vec_t>
int get_partition(vec_t * data,int size,vec_t min,vec_t max,int * hist,int*psum,int hist_size,int myrank)
{
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0, temp_storage_bytes1 = 0;
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
        data, hist, hist_size+1, min, max, size);

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes1, hist, psum, hist_size);
    assert(temp_storage_bytes1 < temp_storage_bytes);
// Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// Compute histograms
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
        data, hist, hist_size+1, min, max, size);
    // cudaFree(d_temp_storage);
    cudaMemset(d_temp_storage, 0, temp_storage_bytes1);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes1, hist, psum, hist_size);
    cudaFree(d_temp_storage);
//printf("device %d CUB psum needs %d aux memory\n",myrank,temp_storage_bytes);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "device %d get partition failed!\n",myrank);
        return 0;
    }
    return 1;
}

#define MASK 0x000000FF
template<typename vec_t>
__global__ static void probe_join_result(vec_t* gpu_dim, vec_t * gpu_dim_index, int* dim_psum, int dim_size, vec_t* gpu_fact,
 vec_t * gpu_fact_index, int* fact_psum,int fact_size, int hsize, vec_t * factFilter,int *pool_ptr,int *gpu_count){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = blockDim.x;
    int block_stride = gridDim.x;

    vec_t dim_data;
    vec_t dim_index;
    vec_t tmp_key[5];
    vec_t tmp_index[5];

    __shared__ vec_t fact_data[300];
    __shared__ vec_t fact_data_index[300];
    __shared__ int HT[256];

    typedef cub::BlockReduce<int, 256> BlockReduce;
    typedef cub::BlockScan<int, 256> BlockScan;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ typename BlockScan::TempStorage temp_storage1;

    int inner = 0;
    int fact_num;
    int count = 0;
    int thread_data;
    int block_aggregate;

    for (int i = bid; i<hsize; i += block_stride){  

        for(int j = tid; j < 300; j += stride)
        {
            fact_data[j] = MAX_INT;
            if(j < 256)
                HT[j] = 0;
        }
        
        __syncthreads();
        int dim_head = dim_psum[i];
        int dim_tail = i == hsize -1 ? dim_size : dim_psum[i+1];
        int fact_head = fact_psum[i];
        int fact_tail = i == hsize -1 ? fact_size : fact_psum[i+1];

        fact_num = fact_tail - fact_head > 300 ? 300 : fact_tail - fact_head;
        inner = 0;
        for(int j = tid + fact_head; j< fact_num + fact_head; j+= stride)
        {
            assert(j < fact_size);
            tmp_key[inner] = gpu_fact[j];
            tmp_index[inner] = gpu_fact_index[j];
            int key = tmp_key[inner] & MASK ;
            atomicAdd(&HT[key],1);
            inner ++;
        }
        __syncthreads();

        thread_data = HT[threadIdx.x];

        __syncthreads();

        BlockScan(temp_storage1).ExclusiveSum(thread_data, thread_data, block_aggregate);

        __syncthreads();

        HT[threadIdx.x] = thread_data;

        __syncthreads();

        for(int j = 0; j < inner; j++)
        {
            int key = tmp_key[j] & MASK ;
            int pos = atomicAdd(&HT[key],1);
            fact_data[pos] = tmp_key[j];
            fact_data_index[pos] = tmp_index[j];
        }

        __syncthreads();

        inner = 0;
        for(int j = tid + dim_head; j< dim_tail; j+= stride)
        {
            assert(j < dim_size);
            dim_data = gpu_dim[j];
            dim_index = gpu_dim_index[j];
            int key = dim_data & MASK;
            int search_num = key == 0? HT[0] : HT[key] - HT[key - 1];
            int pos = key == 0? 0 : HT[key - 1];
            for(int z = 0; z < search_num; z++)
                if(fact_data[pos + z] == dim_data)
                    count += 3;
        }
        __syncthreads();      
    }

    int aggregate = BlockReduce(temp_storage).Sum(count);
    __syncthreads();
    
    if(threadIdx.x == 0)
        atomicAdd(&(*pool_ptr), aggregate); 
}

__device__ int binarySearchLowerBound(int* A, int target, int n){
    int low = 0, high = n, mid;
    while(low <= high){
        mid = low + (high - low) / 2;
        if(target <= A[mid]){
            high = mid - 1;
        }else{
            low = mid + 1;
        }
    }
    if(low < n && A[low] == target)
        return low;
    else
        return -1;
}

template<typename vec_t>
__global__ static void merge_join_result(vec_t* gpu_dim, vec_t * gpu_dim_index, vec_t * bucket, int* dim_psum, int dim_size, vec_t* gpu_fact, 
	vec_t * gpu_fact_index, int* fact_psum,int fact_size, int hsize, int * factFilter,int *gpu_count){

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int stride = blockDim.x;
	int block_stride = gridDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_num = stride * block_stride;
	__shared__ vec_t dim_data[2][512];
	// __shared__ vec_t dim_index[512];
	vec_t fact_data;
	vec_t fact_data_index;
	int inner = 0;
	int count = 0;
	int head,tail;

	for (int i = bid; i<hsize; i += block_stride){      
		for(int j=tid;j<512;j+=stride)
			dim_data[0][j] = MAX_INT;
		__syncthreads();

		int dim_head = dim_psum[i];
		int dim_tail = i == hsize -1 ? dim_size : dim_psum[i+1];
		int fact_head = fact_psum[i];
		int fact_tail = i == hsize -1 ? fact_size : fact_psum[i+1];
		assert(dim_tail - dim_head < 512);
		  //printf("bid %d tid %d head %d tail %d\n", i,tid,fact_head,fact_tail);
		inner = 0;
		for(int j = tid + dim_head; j< dim_tail; j+= stride)
		{
			dim_data[0][tid + inner] = bucket[j * 2];
			dim_data[1][tid + inner] = bucket[j * 2 + 1];
			inner += stride;
		}
		__syncthreads();

		for(int j = tid + fact_head; j< fact_tail; j+= stride)
		{
			fact_data = gpu_fact[j];
			fact_data_index = gpu_fact_index[j];

			// int re = binarySearchLowerBound(dim_data,fact_data,dim_tail - dim_head);
			// if(re != -1)
			// {
			// 	int fact_id = fact_data_index;
			// 	factFilter[fact_id] = dim_index[re];
			// 	count ++;
			// 	atomicAdd(&gpu_count[fact_id % thread_num],1);
			// }

			for(int k = 0; k < dim_tail - dim_head; k++)	
				if(fact_data == dim_data[0][k])
				{
					int fact_id = fact_data_index;
					factFilter[fact_id] = dim_data[1][k];
					count ++;
					atomicAdd(&gpu_count[fact_id % thread_num],1);
					break;
				}
		}
		__syncthreads();	
	}
	// gpu_count[offset] = count;
}


template<typename vec_t>
__global__ static void gpudb_like_probe(vec_t* gpu_dim, vec_t * gpu_dim_index, vec_t * bucket, int * num, int* dim_psum, int dim_size, vec_t* gpu_fact, 
	vec_t * gpu_fact_index, int fact_size, int hsize, int * factFilter,int *count){

	int lcount = 0;
    int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<fact_size;i+=stride){

        int fkey = ((int *)(gpu_fact))[i];
        int hkey = fkey &(hsize-1);
        int keyNum = num[hkey];
        int fvalue = 0;

        for(int j=0;j<keyNum;j++){
            int pSum = dim_psum[hkey];
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

template<typename vec_t>
__global__ static void merge_join_result_reuse(vec_t* gpu_dim, vec_t * gpu_dim_index, int* dim_psum, int dim_size, vec_t* gpu_fact, 
	vec_t * gpu_fact_index, int* fact_psum,int fact_size, int hsize, vec_t * factFilter,int *pool_ptr,vec_t *batch1,int num1,vec_t * batch2,uint num2,
	cub::GridBarrier global_barrier,int * flags, int * mutex, int * end,int buffer_size){

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int stride = blockDim.x;
	int block_stride = gridDim.x;
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int reg_block = (hsize / block_stride + (hsize % block_stride == 0 ? 0 : 1)) * block_stride;
	vec_t dim_data[10];
	vec_t dim_index[10];
	__shared__ vec_t fact_data[400];
	__shared__ vec_t fact_data_index[400];
	int inner = 0;
	int count = 0;
	int pos ;
	
	for (int i = bid; i < reg_block; i += block_stride){   
		pos = 0; 
		if(i < hsize)  
		{
			for(int j=tid;j<400;j+=stride)
				fact_data[j] = MAX_INT;
			__syncthreads();

			int dim_head = dim_psum[i];
			int dim_tail = i == hsize -1 ? dim_size : dim_psum[i+1];
			int fact_head = fact_psum[i];
			int fact_tail = i == hsize -1 ? fact_size : fact_psum[i+1];
			if(fact_tail - fact_head > 400 && tid == 0)
				printf("bid %d tid %d head %d tail %d\n", i,tid,fact_head,fact_tail);
			inner = 0;
			for(int j = tid + fact_head; j< fact_tail; j+= stride)
			{
				fact_data[tid + inner] = gpu_fact[j];
				fact_data_index[tid + inner] = gpu_fact_index[j];
				inner += stride;
			}
			__syncthreads();
			inner = 0;
			for(int j = tid + dim_head; j< dim_tail; j+= stride)
			{
				dim_data[inner] = gpu_dim[j];
				dim_index[inner++] = gpu_dim_index[j];
			}

			int pre = 0; 
			for(int z=0;z<inner;z++)
			{
			    int high = 400;
			    int low = 0;
			    while(low <= high )
			    {
			    	count ++;
			        int mid = (high + low)/2;
			        assert(mid < 400);
			        if(fact_data[mid] < dim_data[z])
			            low = mid + 1;
			        else if(fact_data[mid] > dim_data[z])
			            high = mid - 1;
			        else
			        {
			            while(mid-1>=0 && fact_data[mid-1] == dim_data[z])
			            {mid--;}
			            pre = mid;
			            break;                  
			        }
			    }
			    for(int j = pre; j < 400; j++)  
			        if(fact_data[j] == dim_data[z]) 
			        {
			        	count ++;
						pos = atomicAdd(&(*pool_ptr), 3);
						if(pos < num1) batch1[pos] = dim_data[z]; else batch2[pos - num1] = dim_data[z];
						if(pos + 1 < num1) batch1[pos + 1] = fact_data_index[z]; else batch2[pos + 1 - num1] = fact_data_index[z];
						if(pos + 2 < num1) batch1[pos + 2] = dim_index[z]; else batch2[pos + 2 - num1] = dim_index[z];
						// factFilter[pos] = dim_data[z];
						// factFilter[pos+1] = fact_data_index[j];
						// factFilter[pos+2] = dim_index[z];
			        }
			        else
			            break;
			}
		}
		global_barrier.Sync();
		if(pos != 0 && (pos / 3) % buffer_size == 0 || i == hsize - 1)
		{
			// while(atomicCAS(&*mutex,0,1) != 0)
   //          {}
            atomicAdd(&*flags,1);
   //          int f = atomicExch(&*mutex,0);
			// assert(f != 0);
		}
		__syncthreads();
		if(threadIdx.x == 0 && i == hsize - 1)
			(*end) = 0;
	}
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

__global__ static void index_join_result(int* num, int* psum, int* data, int * index, int* fact, long inNum, int* count, int * factFilter,int hsize){

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
            int dimKey = ((int *)(data))[j + pSum];

            if(dimKey == fkey){

                int dimId = ((int *)(index))[j + pSum];
                lcount ++;
                fvalue = dimId + 1;

                break;
            }
        }
        factFilter[i] = fvalue;
    }
    count[offset] = lcount;
}

__global__ static void materialization_index(int* dim, int * dim_attr, int* psum, int* fact, int* fact_attr, long inNum, int * factFilter, int * result1, int * result2, int * result3)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;
    int localCount = psum[offset];

    for(int i=offset;i<inNum;i+=stride){
    	int dimID = factFilter[i];
    	if(dimID != 0)
    	{    		
    		((int*)result1)[localCount] = ((int *)fact)[i];
    		((int*)result2)[localCount] = ((int *)fact_attr)[i];
    		((int*)result3)[localCount] = ((int *)dim_attr)[dimID];
            localCount ++;
    	}
    }
}

__global__ static void materialization(int* dim, int * dim_attr, int* psum, int* fact, int* fact_attr, long inNum, int * factFilter, int * result1, int * result2, int * result3)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;
    int localCount = psum[offset];

    for(int i=offset;i<inNum;i+=stride){
    	int dimID = factFilter[i];
    	if(dimID != 0)
    	{    		
    		((int*)result1)[localCount] = ((int *)fact)[i];
    		((int*)result2)[localCount] = ((int *)fact_attr)[i];
    		((int*)result3)[localCount] = ((int *)dim_attr)[dimID - 1];
            localCount ++;
    	}
    }
}

__global__ static void scanCol(int inNum, int * result1, int * result2, int * result3, bool *mapbit)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
    	if(result2[i] == result3[i])
    	{
    		mapbit[i] = true;
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


__global__ static void build_hash_table_nest(int *dim, int * dim_index, long inNum, int *psum, int * bucket,int hsize){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
        int joinKey = ((int *) dim)[i]; 
        int joinAttr = ((int *) dim_index)[i]; 
        int hKey = joinKey & (hsize-1);
        int pos = atomicAdd(&psum[hKey],1) * 2;
        assert(pos < inNum * 3);
        ((int*)bucket)[pos] = joinKey;
        pos += 1;
        ((int*)bucket)[pos] = joinAttr;
    }
}

__global__ static void count_join_result_nest(int* num, int* psum, int* bucket, int* fact, int * fact_attr, long inNum, bool* count,int hsize){

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
                int dimAttr = ((int *)(bucket))[2*j + 2*pSum + 1];
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
template<typename vec_t>
__global__ static void reorganize(vec_t * key, vec_t * index, int inNum, vec_t * bucket)
{
	int stride = blockDim.x * gridDim.x;
    long offset = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=offset;i<inNum;i+=stride){
    	bucket[i*2] = key[i];
    	bucket[i*2+1] = index[i];
    }
}



__global__ static void nestScanFilterDistributedCorrelated_deviceFunc(//Sofokils function
     int *col, int * innerTableSecondLinkingPredicate_d,  long tupleNum, // Inner table
     int * outerTableData, int * outerTableSecondLinkingPredicate_d, int outerTableDataRows, //Outer Table
     int * num, int * psum, int hsize,
     int innerThreadGpuMapping, //Inner mapping
     bool * innerOuterMatchingBitmap) //Output
    {

    //Need to think how we are going to compare the second predicate

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con, outerValue, outerValue2, innerValue, innerValue2;
    int indexKey, indexNum, indexPos ;


    if(innerThreadGpuMapping == 0) // Sofoklis default mapping
    {
	    for (long j=0; j<outerTableDataRows;j++){
	        
		    /* Get outer table values */
		    outerValue = outerTableData[j];
		    outerValue2 = outerTableSecondLinkingPredicate_d[j];
		    
		    /* Filter result */
		    for(long i = tid; i<tupleNum;i+=stride){
		    
		        //Get inner values
		        innerValue = ((int*)col)[i];
		        innerValue2 = ((int*)innerTableSecondLinkingPredicate_d)[i];

		        /* Store bool value (only threads that have a match) */
		        if (outerValue == innerValue && outerValue2 == innerValue2){
		            innerOuterMatchingBitmap[j]=true;
		            break;
		        }
		    }
		}
	}
	else //sort index scan
	{
		for (long j=tid; j<outerTableDataRows;j+=stride){	        
		    /* Get outer table values */
		    outerValue = outerTableData[j];
		    outerValue2 = outerTableSecondLinkingPredicate_d[j];

		    indexKey   = outerValue & (hsize -1);
		    indexNum   = num[indexKey];
		    indexPos   = psum[indexKey];

		    /* Filter result */
		    for(long i = 0; i<indexNum;i++){
		    
		        //Get inner values
		        innerValue = ((int*)col)[i + indexPos];
		        innerValue2 = ((int*)innerTableSecondLinkingPredicate_d)[i + indexPos];

		        /* Store bool value (only threads that have a match) */
		        if (outerValue == innerValue && outerValue2 == innerValue2){
		            innerOuterMatchingBitmap[j]=true;
		            break;
		        }
		    }
		}
	}
}


__global__ static void unnestScanFilterDistributedCorrelated_deviceFunc(
     int *col, int * innerTableSecondLinkingPredicate_d,  long tupleNum, // Inner table
     int * index, // index for inner table
     int * outerTableData, int * outerTableSecondLinkingPredicate_d, int outerTableDataRows, //Outer Table
     int * num, int * psum, int hsize,
     int innerThreadGpuMapping, //Inner mapping
     bool * innerOuterMatchingBitmap) //Output
    {

    //Need to think how we are going to compare the second predicate

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con, outerValue, outerValue2, innerValue, innerValue2;
    int indexKey, indexNum, indexPos ;

    if(innerThreadGpuMapping == 0)
    {
	    for (long j=0; j<outerTableDataRows;j++){
	        
		    /* Get outer table values */
		    outerValue = outerTableData[j];
		    outerValue2 = outerTableSecondLinkingPredicate_d[j];
		    
		    /* Filter result */
		    for(long i = tid; i<tupleNum;i+=stride){
		    
		        //Get inner values
		        innerValue = ((int*)col)[i];
		        innerValue2 = ((int*)innerTableSecondLinkingPredicate_d)[i];

		        /* Store bool value (only threads that have a match) */
		        if (outerValue == innerValue && outerValue2 == innerValue2){
		            innerOuterMatchingBitmap[j]=true;
		            break;
		        }
		    }
		}
	}
	else
	{
		for (long j=tid; j<outerTableDataRows;j+=stride){	        
		    /* Get outer table values */
		    outerValue = outerTableData[j];
		    outerValue2 = outerTableSecondLinkingPredicate_d[j];

		    indexKey   = outerValue & (hsize -1);
		    indexNum   = num[indexKey];
		    indexPos   = psum[indexKey];

		    /* Filter result */
		    for(long i = 0; i<indexNum;i++){
		    	int target = index[i + indexPos];
		        //Get inner values
		        innerValue = ((int*)col)[target];
		        innerValue2 = ((int*)innerTableSecondLinkingPredicate_d)[target];

		        /* Store bool value (only threads that have a match) */
		        if (outerValue == innerValue && outerValue2 == innerValue2){
		            innerOuterMatchingBitmap[j]=true;
		            break;
		        }
		    }
		}
	}
}

__global__ static void recallScan(
     int * col, int * innerTableSecondLinkingPredicate_d, int tupleNum,// Inner table
     int outerValue, int outerValue2, int offset, //Outer Table
     bool * innerOuterMatchingBitmap) //Output
{

	int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con, innerValue, innerValue2;

	for(long i = tid; i<tupleNum;i+=stride){
		    
        //Get inner values
        innerValue = ((int*)col)[i];
        innerValue2 = ((int*)innerTableSecondLinkingPredicate_d)[i];

        /* Store bool value (only threads that have a match) */
        if (outerValue == innerValue && outerValue2 == innerValue2){
            innerOuterMatchingBitmap[offset]=true;
            break;
        }
    }
} 

__global__ static void nestScanDynamic_deviceFunc(
     int *col, int * innerTableSecondLinkingPredicate_d,  long tupleNum, // Inner table
     int * outerTableData, int * outerTableSecondLinkingPredicate_d, int outerTableDataRows, //Outer Table
     int innerThreadGpuMapping, //Inner mapping
     bool * innerOuterMatchingBitmap) //Output
    {

    //Need to think how we are going to compare the second predicate

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con, outerValue, outerValue2, innerValue, innerValue2;
    int indexKey, indexNum, indexPos ;

    if(innerThreadGpuMapping == 0)
    {
	    for (long j=tid; j<outerTableDataRows;j+=stride){
	        
		    /* Get outer table values */
		    outerValue = outerTableData[j];
		    outerValue2 = outerTableSecondLinkingPredicate_d[j];

		    recallScan<<<4096,256>>>((int*)col,(int*)innerTableSecondLinkingPredicate_d,tupleNum,outerValue,outerValue2,j,innerOuterMatchingBitmap);
		    
		    __syncthreads();
		    // /* Filter result */
		    // for(long i = tid; i<tupleNum;i+=stride){
		    
		    //     //Get inner values
		    //     innerValue = ((int*)col)[i];
		    //     innerValue2 = ((int*)innerTableSecondLinkingPredicate_d)[i];

		    //     /* Store bool value (only threads that have a match) */
		    //     if (outerValue == innerValue && outerValue2 == innerValue2){
		    //         innerOuterMatchingBitmap[j]=true;
		    //         break;
		    //     }
		    // }
		}
		// cudaDeviceSynchronize();
	}
}

__global__ static void recallScanUmlimited(
     int outerValue, int * allCol, int innerIndex, int outerIndex, int currentLevel, 
     int loopLevel, int * attrSize, int * attrNum, int * round, bool * innerOuterMatchingBitmap) //Output
{

	int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con, innerValue;

    //Get inner values
    int index = innerIndex - attrNum[0];
    int tableIndex, tableBegin;
    if(index >= 2)
    {
    	index -= 2;
    	tableIndex = 2;
    	tableBegin = attrNum[0] * attrSize[0] + attrNum[1] * attrSize[1];
    }
    else
    {
    	tableIndex = 1;
    	tableBegin = attrNum[0] * attrSize[0];
    }
    int tupleNum = attrSize[tableIndex];

	for(long i = tid; i<tupleNum;i+=stride){		    
        
        innerValue = allCol[tableBegin + tupleNum * index + i];

        /* Store bool value (only threads that have a match) */
        if (outerValue == innerValue){
        	if(currentLevel == loopLevel)
        	{
            	innerOuterMatchingBitmap[outerIndex] = true;
            	return;
        	}
            else
            {
            	int inner1 = round[currentLevel * 2 + 1];
            	int outer1 = round[currentLevel * 2];

            	int outerValue1 = allCol[outer1 * attrSize[0] + outerIndex];
            	recallScanUmlimited<<<4096,256>>>(outerValue1, allCol, inner1, outerIndex, currentLevel + 1, loopLevel, attrSize, attrNum, round,innerOuterMatchingBitmap);
            }
        }
    }
} 


__global__ static void nestScanDynamicUnlimited_deviceFunc(
     int * allCol, int * nuknow, int loopLevel, int * attrSize, int * attrNum, int * round,
     int innerThreadGpuMapping, //Inner mapping
     bool * innerOuterMatchingBitmap) //Output
    {

    //Need to think how we are going to compare the second predicate

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con, outerValue, outerValue2, innerValue, innerValue2;
    int indexKey, indexNum, indexPos ;

    if(innerThreadGpuMapping == 0)
    {
	    for (long j=tid; j<attrSize[0];j+=stride){
	        
	        int r0 = round[0];
		    /* Get outer table values */
		    outerValue = allCol[r0 * attrSize[0] + j];
		    int r1 = round[1];
		    recallScanUmlimited<<<2048,128>>>(outerValue, allCol, r1, j, 1, loopLevel, attrSize, attrNum, round,innerOuterMatchingBitmap);
		    
		    __syncthreads();

		}
	}
}

cudaError_t Subquery_proc(int *dim,int *dim_attr,int *dim_index,int lsize,int *fact,int *fact_attr,int* fact_index,int rsize)
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
    int hsize = lsize;
    NP2(hsize);

    int *gpu_dim_psum = NULL, *gpu_fact_psum = NULL, *gpu_dim_hashNum = NULL;
    int total_count = 0;
    int all_count = 0;
    int * d_max, max, mask,*gpu_count_db = NULL,*gpu_count_new = NULL;
    int alloc_size = rsize > lsize ? rsize : lsize;
    int * factFilter_db = NULL, * bucket = NULL; 
    int * filterResult1 = NULL, * filterResult2 = NULL, * filterResult3 = NULL, * gpu_resPsum = NULL;
    bool * mapbit =NULL;
    clock_gettime(CLOCK_REALTIME,&start_t);
    
    CUDACHECK(cudaMalloc((void **)&gpu_dim_hashNum, hsize * sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&gpu_dim_psum, hsize * sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&gpu_fact_psum, hsize * sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&d_max, sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&gpu_count_db, 4096*256*sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&gpu_resPsum, 4096*256*sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&factFilter_db, rsize * sizeof(int))); 
    CUDACHECK(cudaMemset(gpu_dim_psum,0, hsize * sizeof(int)));
    CUDACHECK(cudaMemset(gpu_fact_psum,0, hsize * sizeof(int)));

    CUDACHECK(cudaMalloc((void **)&mapbit, lsize * sizeof(bool)));    

    CUDACHECK(cudaMalloc((void **)&bucket, lsize * 3 * sizeof(int))); 

    cudaMemset(mapbit, 0, lsize * sizeof(bool));

    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("init Time: %lf ms hsize %d full block size %d\n", timeE/(1000*1000),hsize,grid.x);
    

    clock_gettime(CLOCK_REALTIME,&start_t);
    CUDACHECK(cudaMemset(gpu_dim_hashNum,0, hsize * sizeof(int)));  
    CUDACHECK(cudaMemset(factFilter_db,0, rsize * sizeof(int)));    
    CUDACHECK(cudaMemset(gpu_count_db,0, 4096*256 * sizeof(int)));   
    
    count_hash_num<<<4096,256>>>(dim,lsize,gpu_dim_hashNum,hsize);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::exclusive_scan(thrust::device, gpu_dim_hashNum, gpu_dim_hashNum + hsize, gpu_dim_psum); // in-place scan

    cudaMemcpy(gpu_fact_psum, gpu_dim_psum, hsize * sizeof(int), cudaMemcpyDeviceToDevice);

    build_hash_table<<<4096,256>>>(dim,lsize,gpu_fact_psum,bucket,hsize);

    CUDACHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("build Time: %lf ms \n",  timeE/(1000*1000));
    total += timeE;   

    clock_gettime(CLOCK_REALTIME,&start_t);

    count_join_result<<<4096,256>>>(gpu_dim_hashNum,gpu_dim_psum,bucket,fact,rsize,gpu_count_db,factFilter_db,hsize);
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

    materialization<<<4096,256>>>(dim, dim_attr, gpu_resPsum, fact, fact_attr, rsize, factFilter_db, filterResult1,filterResult2,filterResult3);
    CUDACHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("materialization Time: %lf ms \n",  timeE/(1000*1000));
    total += timeE;  

    clock_gettime(CLOCK_REALTIME,&start_t);

    scanCol<<<4096,256>>>(resCount, filterResult1,filterResult2,filterResult3,mapbit);

    CUDACHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_REALTIME,&end_t);
    timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
    printf("scan Time: %lf ms \n", timeE/(1000*1000));
    total += timeE;   

    cudaFree(filterResult1);
    cudaFree(filterResult2);
    cudaFree(filterResult3);

    printf("GPU-DB unnest Time: %lf ms\n\n", total/(1000*1000));

    int * allAttr = NULL, * roundAttr = NULL, * d_attrSize = NULL, * d_attrNum = NULL, * d_roundNum;
    //three table size
    int attrSize[3] = {lsize, rsize, rsize};
    //number of attrs in each table
    int attrNum[3] = {3, 2, 2};    
    int roundNum[6] = {0, 3 , 1, 4, 0, 3};
    // indicate the comaprison , even number only, three loops
    //e.g., the first number of 2 indicate the first two attrs in roundNum will be compared
    int roundHist[3] = {2,2,2}; 
    int roundPsum[3] = {0,2,4};
    // only two loops now
    int roundLevel = 2;

    int attrTotalSize = 0;
    attrTotalSize = attrNum[0] * attrSize[0] + attrNum[1] * attrSize[1] + attrNum[2] * attrSize[2];

    CUDACHECK(cudaMalloc((void **)&allAttr, attrTotalSize * sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&roundAttr, 3 * sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&d_attrSize, 3 * sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&d_attrNum, 3 * sizeof(int))); 
    CUDACHECK(cudaMalloc((void **)&d_roundNum, 6 * sizeof(int))); 

    CUDACHECK(cudaMemcpy(d_attrSize,attrSize, 3 * sizeof(int),cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_attrNum,attrNum, 3 * sizeof(int),cudaMemcpyHostToDevice));

    cudaMemcpy(d_roundNum,roundNum, 6 * sizeof(int),cudaMemcpyHostToDevice); 

    attrTotalSize = 0;

    cudaMemcpy(allAttr,dim, lsize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(allAttr + lsize,dim_attr, lsize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(allAttr + 2 * lsize,dim_index, lsize * sizeof(int), cudaMemcpyDeviceToDevice);

    attrTotalSize += attrNum[0] * attrSize[0];

    cudaMemcpy(allAttr + attrTotalSize,fact, rsize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(allAttr + attrTotalSize + rsize,fact_attr, rsize * sizeof(int), cudaMemcpyDeviceToDevice);

    attrTotalSize += attrNum[1] * attrSize[1];

    cudaMemcpy(allAttr + attrTotalSize,fact_index, rsize * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(allAttr + attrTotalSize + rsize,fact, rsize * sizeof(int), cudaMemcpyDeviceToDevice);    

    clock_gettime(CLOCK_REALTIME,&start_t);

    cudaMemset(mapbit, 0, lsize * sizeof(bool));

    nestScanDynamicUnlimited_deviceFunc<<<32,128>>>(allAttr,roundAttr,roundLevel,d_attrSize,d_attrNum,d_roundNum,0,mapbit);

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


cudaError_t Preparation(int *all_l_table, int *all_l_attr, int l_size, int *all_r_table, int *all_r_attr, int r_size, int argc, char* argv[]);

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
	int *a,*a_1;
	int *b,*b_1;

	fprintf(stderr,"R relation size %d Rows , S relation size %d Rows bit %d\n",a_size,b_size);

	CUDACHECK(cudaHostAlloc((void **)&a, sizeof(int)*a_size, cudaHostAllocPortable | cudaHostAllocMapped));
	CUDACHECK(cudaHostAlloc((void **)&b, sizeof(int)*b_size, cudaHostAllocPortable | cudaHostAllocMapped));
	CUDACHECK(cudaHostAlloc((void **)&a_1, sizeof(int)*a_size, cudaHostAllocPortable | cudaHostAllocMapped));
	CUDACHECK(cudaHostAlloc((void **)&b_1, sizeof(int)*b_size, cudaHostAllocPortable | cudaHostAllocMapped));

	for (int i = 0; i < b_size; i++)
	{
		int tmp = rand()%a_size;
		if(!tmp)
			tmp ++;
		b[i] = tmp;	
		b_1[i] = b_size - i;	
	}
	for (int i = 0; i < a_size; i++)
	{
		int tmp = rand()%a_size;
		if(!tmp)
			tmp ++;
		a[i] = tmp;
		a_1[i] = a_size - i;	
	}
	cudaError_t cudaStatus;

	cudaStatus = Preparation(a, a_1, a_size,b, b_1, b_size, argc, argv);

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

cudaError_t Preparation(int *all_l_table, int *all_l_index, int l_size, int *all_r_table, int *all_r_index, int r_size, int argc, char* argv[])
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
	int * gpu_dim     = NULL, * gpu_dim_index = NULL, * gpu_fact  = NULL, *gpu_fact_index = NULL;
	int * gpu_dim_attr= NULL, * gpu_fact_attr = NULL;

	clock_gettime(CLOCK_REALTIME,&start_t);
	int primaryKeySize = r_size * sizeof(int);
	int filterSize = l_size * sizeof(int);

	CUDACHECK(cudaMalloc((void **)&gpu_fact, primaryKeySize));
	CUDACHECK(cudaMalloc((void **)&gpu_fact_attr, primaryKeySize));
	CUDACHECK(cudaMalloc((void **)&gpu_dim, filterSize));
	CUDACHECK(cudaMalloc((void **)&gpu_dim_attr, filterSize));
	CUDACHECK(cudaMalloc((void **)&gpu_fact_index, primaryKeySize));
	CUDACHECK(cudaMalloc((void **)&gpu_dim_index, filterSize));

	clock_gettime(CLOCK_REALTIME,&end_t);
	double timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
	total += timeE;

	printf("init Time: %lf ms\n", timeE/(1000*1000));	

	clock_gettime(CLOCK_REALTIME,&start_t);
	CUDACHECK(cudaMemcpyAsync(gpu_dim, all_l_table, sizeof(int)*l_size, cudaMemcpyHostToDevice, s));
	CUDACHECK(cudaMemcpyAsync(gpu_fact, all_r_table , sizeof(int)*r_size, cudaMemcpyHostToDevice, s));
	CUDACHECK(cudaMemcpyAsync(gpu_dim_attr, all_l_index, sizeof(int)*l_size, cudaMemcpyHostToDevice, s));
	CUDACHECK(cudaMemcpyAsync(gpu_dim_attr, all_r_index , sizeof(int)*r_size, cudaMemcpyHostToDevice, s));
	assign_index<int><<<grid,block>>>(gpu_dim_index,l_size,0);
	assign_index<int><<<grid,block>>>(gpu_fact_index,r_size,0);
	CUDACHECK(cudaDeviceSynchronize());

	clock_gettime(CLOCK_REALTIME,&end_t);
	timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
	total += timeE;
	int recv_offset = 0;

	printf("Host To Device Time: %lf ms\n", timeE/(1000*1000));

	clock_gettime(CLOCK_REALTIME, &start_t);

	cudaStatus = Subquery_proc(gpu_dim,gpu_dim_attr,gpu_dim_index,l_size,gpu_fact,gpu_dim_attr,gpu_fact_index,r_size);
	
	clock_gettime(CLOCK_REALTIME,&end_t);
	timeE = (end_t.tv_sec -  start_t.tv_sec)* BILLION + end_t.tv_nsec - start_t.tv_nsec;
	total += timeE;

	printf("Subquery Time: %lf ms\n", timeE/(1000*1000));
	
	clock_gettime(CLOCK_REALTIME, &start_t);
	CUDACHECK(cudaFree(gpu_fact));
	CUDACHECK(cudaFree(gpu_fact_index));
	CUDACHECK(cudaFree(gpu_dim));
	CUDACHECK(cudaFree(gpu_dim_index));

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