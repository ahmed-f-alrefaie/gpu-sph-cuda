// cuda_example3.cu : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
// CUDA runtime
#include <cuda_runtime.h>

void incrementArrayOnHost(float *a, int N)
{
	int i;
	for(i=0; i<N; i++) a[i]+1.f;
};

__global__ void incrementArrayOnDevice(float *a, int N)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < N) a[idx] = a[idx]+1.f;
};
/*
int main(void)
{
	float *a_h, *b_h;
	float *a_d;
	int i, N=10;
	size_t size = N*sizeof(float);
	//Allocate arrays on the host
	a_h = (float*)malloc(size);
	b_h = (float*)malloc(size);
	//allocate array on device
	cudaMalloc((void **) &a_d, size);
	//Initialize the data
	for(i = 0; i<N; i++) a_h[i] = (float)i;
	cudaMemcpy(a_d,a_h, sizeof(float)*N, cudaMemcpyHostToDevice);
	//Do calculation
	incrementArrayOnHost(a_h, N);
	//Do calculation on device
	//Figure out execution config
	int blockSize = 4;
	int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
	//Call on the device
	incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_d, N);
	//Retrive the result and store in b_h
	cudaMemcpy(b_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
	//check results
	for(int i = 0; i < N; i++) assert(a_h[i]==b_h[i]);
	//cleanup
	free(a_h); 
	free(b_h); 
	cudaFree(a_d);
}
*/