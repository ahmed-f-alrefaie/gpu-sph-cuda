#include "SPHSolverHost.cuh"
#include "SPHSolverKernal.cu"

#include <thrust\sort.h>
#include "helper_cuda.h"

#pragma once

#define FLUID_SIZE 72;

#define PI 3.141512

__device__ char* partBuffer; //Data array for particles
__device__ uint* hashBuffer; //unsorted hash table
__device__ uint* hashValueBuffer;
__device__ char** sortedPartBuffer; //sorted particle array
__device__ int2* gridBuffer;


SolverParams params;

uint nearest_pow (uint num)
{
    uint n = num > 0 ? num - 1 : 0;

    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;

    return n;
};


int2* clearBuffer;


void cudaInit(int argc,const char **argv)
{
	//Firse we find the best device
	int deviceId = findCudaDevice(argc, argv);
	if(deviceId != -1)
	{
	//Initialise Cuda with best device
	gpuDeviceInit(deviceId);
	}else
		printf("\n No CUDA compatible device found");

	cudaDeviceProp p;
	cudaGetDeviceProperties ( &p, 0);
	
	printf ( "-- CUDA --\n" );
	printf ( "Name:       %s\n", p.name );
	printf ( "Revision:   %d.%d\n", p.major, p.minor );
	printf ( "Global Mem: %d\n", p.totalGlobalMem );
	printf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	printf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	printf ( "Warp Size:  %d\n", p.warpSize );
	printf ( "Mem Pitch:  %d\n", p.memPitch );
	printf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	printf ( "Const Mem:  %d\n", p.totalConstMem );
	printf ( "Clock Rate: %d\n", p.clockRate );
	printf ( "Max Grid Size: %d\n",p.maxGridSize);

	//Used to compute number of blocks and threads needed
	params.maxThreadsPerBlock =//256
		p.maxThreadsPerBlock;
	


}

void ComputeThreadsAndBlocks(int numPoints,int &numBlocks,int &numThreads)
{
	numThreads = min(params.maxThreadsPerBlock,numPoints);
	//Calculate how many blocks we need
	numBlocks = numPoints/numThreads;
		if(numPoints%numThreads != 0)numBlocks++;
}


void SetupCUDA(int partCount,int stride)
{
	

	//Compute the needed blocks and threads
	ComputeThreadsAndBlocks(partCount,params.numBlocks,params.numThreads);
	params.szPnts = params.numBlocks*params.numThreads*stride;
	params.szHash = params.numBlocks*params.numThreads*sizeof(uint);
	params.bytestride = stride;

	params.fluidnum = partCount;
	//Allocate memory in the device for the fluid particles
	checkCudaErrors(cudaMalloc( (void**)&partBuffer,  params.szPnts)) ;
	checkCudaErrors(cudaMalloc( (void**)&sortedPartBuffer, params.numBlocks*params.numThreads*sizeof(int))) ;
	checkCudaErrors(cudaMalloc( (void**)&hashBuffer, params.szHash)) ;
	checkCudaErrors(cudaMalloc( (void**)&hashValueBuffer, params.szHash)) ;
	//checkCudaErrors(cudaMalloc( (void**)&sortedHashBuffer, params.szHash)) ;
//	checkCudaErrors(cudaMalloc( (void**)&solver, sizeof(SolverParams))) ;

	params.simSize = make_float3(1,1,1);
	
//	checkCudaErrors(cudaGetSymbolAddress( &address, "solver"));

	//Copy the information into the constant in the device
	checkCudaErrors(cudaMemcpyToSymbol(solver,&params,sizeof(params),0, cudaMemcpyHostToDevice));

	cudaThreadSynchronize(); //Ensure this is done before continuing
}


void SetupSolver(float smoothRadius, int numFluid)
{
	params.smoothradius = smoothRadius;
	params.gridSize = smoothRadius;

	params.hashNum = nearest_pow(numFluid); //find nearest power of 2

	checkCudaErrors(cudaMalloc( (void**)&gridBuffer,params.hashNum*sizeof(uint2)));
	//create the clear buffer
	clearBuffer = new int2[params.hashNum];
	for(int i = 0; i < params.hashNum; i++)
		clearBuffer[i] = make_int2(-1,-1);
	//calculate the kernals
	params.kernal = 315/(64*PI*std::powf(params.smoothradius,9));
	params.divKernal = -945/(32*PI*std::powf(params.smoothradius,9));
	params.lapKernal = -945/(32*PI*std::powf(params.smoothradius,9));
	params.pressKernal = -45/(PI*std::powf(params.smoothradius,6));
	params.viscKernal = 45/(PI*std::powf(params.smoothradius,6));
	params.velDamp = -.01f;
	params.maxSpeed = 200.0f;
	//Copy the information into the constant in the device
	checkCudaErrors(cudaMemcpyToSymbol(solver,&params,sizeof(params),0, cudaMemcpyHostToDevice));

	cudaThreadSynchronize(); //Ensure this is done before continuing

}

	void SwapParticleBuffers()
	{
		//std::swap(sortedPartBuffer,partBuffer);
	};

void TestParticleBuffer(char* pData,int numParts)
{
	int i;
	for(i = 0; i < numParts; i++)
	{
		float3* pos = (float3*)(pData + i*params.bytestride);
		{
			printf("Particle %i - x: %f y: %f z: %f\n",i,pos->x,pos->y,pos->z);
		}
	}
}

void ComputeDensityCUDA()
{
	ComputeDensityAndPressure<<< params.numBlocks, params.numThreads>>>(sortedPartBuffer,params.fluidnum,gridBuffer);
	checkCudaErrors(cudaThreadSynchronize()); 
	//ComputeForces<<< params.numBlocks, params.numThreads>>>(sortedPartBuffer,params.fluidnum,gridBuffer);
	//checkCudaErrors(cudaThreadSynchronize());

}

void ComputeForcesCUDA()
{
	ComputeForces<<< params.numBlocks, params.numThreads>>>(sortedPartBuffer,params.fluidnum,gridBuffer);
	checkCudaErrors(cudaThreadSynchronize());
}

void AdvanceSPH(float dt)
{
	AdvanceLeap<<< params.numBlocks, params.numThreads>>>(dt, sortedPartBuffer,params.fluidnum);
	cudaThreadSynchronize();



}

void CheckMemory()
{
		size_t avail;
	size_t total;
	//cudaMemGetInfo
	checkCudaErrors(cudaMemGetInfo( &avail, &total ));
	size_t used = total - avail;
	std::cout << "Device memory used: " << used << std::endl;
}


void ClearCUDA()
{
	 checkCudaErrors(cudaFree( partBuffer ) );
	checkCudaErrors( cudaFree( hashBuffer ) );
	checkCudaErrors( cudaFree( hashValueBuffer ) );
	checkCudaErrors( cudaFree( sortedPartBuffer ) );
	checkCudaErrors( cudaFree( gridBuffer));
	delete clearBuffer;
}

void CheckSortedParticles()
{
	char* testparts = (char*)malloc(params.szPnts);
	checkCudaErrors( cudaMemcpy(testparts,partBuffer,params.szPnts, cudaMemcpyDeviceToHost) );
		int i;
		for(i = 0; i < params.fluidnum; i++)
	{
		float3* pos = (float3*)(testparts + i*params.bytestride);
		float* dens = (float*)(testparts + i*params.bytestride + DENS_STRIDE);
		float3* force = (float3*)(testparts + i*params.bytestride + FORCE_STRIDE);
		float* mass = (float*)(testparts + i*params.bytestride + MASS_STRIDE);
		int3 iPos;
		iPos.x = (pos->x/params.gridSize);
		iPos.y = (pos->y/params.gridSize);
		iPos.z = (pos->z/params.gridSize);
		uint hash = ((iPos.x*PRIME_CONSTANT_1)^(iPos.y*PRIME_CONSTANT_2)^(iPos.z*PRIME_CONSTANT_3)) & (params.hashNum-1);

		printf("Hash: %i\n",hash);
		printf("Particle %i - x: %f y: %f z: %f density: %f fx: %f fy: %f fz: %f mass: %f\n",i,pos->x,pos->y,pos->z,*dens,force->x,force->y,force->z,*mass);

		
	}
		free(testparts);
}

void TransferToCUDA(char* pData,int numParts,int stride)
{
	checkCudaErrors( cudaMemcpy(partBuffer,pData,stride*numParts, cudaMemcpyHostToDevice) ); //Transfer the particle data to the CUDA GPU
	cudaThreadSynchronize(); //Ensure this is done before continuing
}

void TransferFromCUDA(char* toData,int numParts,int stride)
{
	cudaThreadSynchronize();
	checkCudaErrors( cudaMemcpy(toData,partBuffer,stride*numParts, cudaMemcpyDeviceToHost) ); //Transfer the particle data to the CUDA GPU
	cudaThreadSynchronize(); //Ensure this is done before continuing
}


void InsertParticles()
{

	checkCudaErrors(cudaMemset ( hashBuffer,0,params.szHash)); //Clear hash
	checkCudaErrors(cudaMemset ( hashValueBuffer,0,params.szHash)); //Clear hash
	checkCudaErrors(cudaMemcpy (gridBuffer,clearBuffer,params.hashNum*sizeof(uint2),cudaMemcpyHostToDevice)); //Clear hash
	//Hash the data

	HashParticles<<< params.numBlocks, params.numThreads>>> (partBuffer,params.fluidnum,hashBuffer,hashValueBuffer);
	cudaThreadSynchronize();

	//Do a radix sort
	thrust::device_ptr<uint> keys(hashBuffer);
	thrust::device_ptr<uint> value(hashValueBuffer);
	//sort the data
	thrust::sort_by_key(keys,keys+( params.numBlocks*params.numThreads),value);
	cudaThreadSynchronize();

	//Sort the particles
	uint smemSize = sizeof(uint)*(params.numThreads+1);
	SortParticles<<<params.numBlocks, params.numThreads,smemSize>>>(partBuffer,sortedPartBuffer, params.fluidnum,hashBuffer,hashValueBuffer, gridBuffer);
	cudaThreadSynchronize();


}

void CheckHashes()
{
	uint* testHashes = (uint*)malloc(params.szHash);
	checkCudaErrors( cudaMemcpy(testHashes,hashBuffer,params.szHash, cudaMemcpyDeviceToHost) );
	uint* testHashesValue = (uint*)malloc(params.szHash);
	checkCudaErrors( cudaMemcpy(testHashesValue,hashValueBuffer,params.szHash, cudaMemcpyDeviceToHost) );
	int i;
	for(i = 0; i <params.fluidnum; i++)
		printf("Hashes - hash: %u fluid id: %i\n",unsigned int((testHashes[i])),(testHashesValue[i]));
	free(testHashes);
	free(testHashesValue);

	int2* gridCheck = (int2*)malloc(params.hashNum*sizeof(int2));
	checkCudaErrors( cudaMemcpy(gridCheck,gridBuffer,params.hashNum*sizeof(uint2), cudaMemcpyDeviceToHost) );
	for(i = 0; i <params.hashNum; i++)
		printf("Cell %i x: %i, y: %i \n",i,gridCheck[i].x,gridCheck[i].y);

};