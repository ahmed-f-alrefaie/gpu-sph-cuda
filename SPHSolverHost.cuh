#include <vector_types.h>	
#include <cuda.h>
#include <driver_types.h>			// for cudaStream_t
#include <iostream>
#pragma once

typedef unsigned int uint;



struct SolverParams
{
	int fluidnum;
	int maxThreadsPerBlock;
	int	numThreads, numBlocks;	
	int	szPnts, szHash;
	int bytestride;
	int hashNum;
	float gridSize;
	float3 simSize;
	float velDamp;
	float maxSpeed;
	float smoothradius,kernal,divKernal,lapKernal;
	float pressKernal,viscKernal;

};



extern "C"
{
	void cudaInit(int argc,const char** argv);
	void TransferToCUDA(char* pData,int count, int stride);
	void SetupCUDA(int partCount,int stride);
	void SetupSolver(float smoothradius, int numParts);
	void InsertParticles();
	void ClearCUDA();
	void CheckHashes();
	void CheckSortedParticles();
	void ComputeDensityCUDA();
	void AdvanceSPH(float dt);
	void TransferFromCUDA(char* toData, int count, int stride);
	void TestParticleBuffer(char* pData,int numParts);
	void CheckMemory();
	void ComputeForcesCUDA();

	void SwapParticleBuffers();
}