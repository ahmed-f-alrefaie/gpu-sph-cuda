#include "SPHSolverHost.cuh"

#pragma once




__global__ void HashParticles(char* pFluidBuf,int numParts,uint2* pHashBuff);