#ifndef SCAN_IMPL_CU
#define SCAN_IMPL_CU

#include "scan.cu"
#include "../include/gpuCudaLib.h"
#include "../include/common.h"

static void scanImpl(int *d_input, int rLen, int *d_output, struct statistic * pp)
{
	int len = 2;
	if(rLen < len){
		int *input, *output;
		cudaMalloc((void**)&input,len*sizeof(int));
		cudaMalloc((void**)&output, len*sizeof(int));
		cudaMemset(input, 0, len*sizeof(int));
		cudaMemcpy(input, d_input, rLen*sizeof(int), cudaMemcpyDeviceToDevice);
		preallocBlockSums(len);
		prescanArray(output, input, len, pp);
		deallocBlockSums();
		cudaMemcpy(d_output,output,rLen*sizeof(int),cudaMemcpyDeviceToDevice);
		cudaFree(input);
		cudaFree(output);
		return;
	}else{

		//Start timer for scanImpl Step 4.2.1 - preallocBlockSums time
		struct timespec startScanImplS1, endScanImplS1;
		clock_gettime(CLOCK_REALTIME,&startScanImplS1);

		preallocBlockSums(rLen);

		//Stop timer for scanImpl Step 4.2.1 - preallocBlockSums time
		CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
		clock_gettime(CLOCK_REALTIME, &endScanImplS1);
		pp->preallocBlockSums_scanImpl_S1 += (endScanImplS1.tv_sec - startScanImplS1.tv_sec)* BILLION + endScanImplS1.tv_nsec - startScanImplS1.tv_nsec;
		

		//Start timer for scanImpl Step 4.2.2 - prescanArray time
		struct timespec startScanImplS2, endScanImplS2;
		clock_gettime(CLOCK_REALTIME,&startScanImplS2);

		prescanArray(d_output, d_input, rLen, pp);

		//Stop timer for scanImpl Step 4.2.2 - prescanArray time
		CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
		clock_gettime(CLOCK_REALTIME, &endScanImplS2);
		pp->prescanArray_scanImpl_S2 += (endScanImplS2.tv_sec - startScanImplS2.tv_sec)* BILLION + endScanImplS2.tv_nsec - startScanImplS2.tv_nsec;
		

		//Start timer for scanImpl 4.2.3 - deallocBlockSums time
		struct timespec startScanImplS3, endScanImplS3;
		clock_gettime(CLOCK_REALTIME,&startScanImplS3);

		deallocBlockSums();

		//Stop timer for scanImpl 4.2.3 - deallocBlockSums time
		CUDA_SAFE_CALL(cudaDeviceSynchronize()); //need to wait to ensure correct timing
		clock_gettime(CLOCK_REALTIME, &endScanImplS3);
		pp->deallocBlockSums_scanImpl_S3 += (endScanImplS3.tv_sec - startScanImplS3.tv_sec)* BILLION + endScanImplS3.tv_nsec - startScanImplS3.tv_nsec;		
	}
}


#endif

