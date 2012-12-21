

#include "SPHSolverHost.cuh"
#include <helper_cuda.h>
#include <helper_math.h>
#include <stdio.h>
#include <math.h>

#pragma once

#define PRIME_CONSTANT_1 73856093
#define PRIME_CONSTANT_2 19349663
#define PRIME_CONSTANT_3 83492791

#define POS_STRIDE 0
#define VEL_STRIDE 12
#define VEL_EVAL_STRIDE 24
#define FORCE_STRIDE 36
#define MASS_STRIDE 48
#define DENS_STRIDE 52
#define VIS_STRIDE 56
#define PRESS_STRIDE 60
#define GAS_STRIDE 64
#define REST_STRIDE 68
#define FLUID_STRIDE 72;
#define EPSILON 0.001
#define NULL_HASH 333333

#define MAX_NEIGHBOURS 80;

__constant__ SolverParams solver;

//Hashing function taken from http://image.diku.dk/projects/media/kelager.06.pdf //
__device__ uint hashFunc(int x, int y, int z)
{

	return ((x*PRIME_CONSTANT_1)^(y*PRIME_CONSTANT_2)^(z*PRIME_CONSTANT_3)) & (solver.hashNum-1);
};

__global__ void HashParticles(char* pFluidBuf,int numParts,uint* pHashBuff,uint* pHashValueBuff)
{
	//Calculate the particle id
	uint Idx = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	//Get the position of the fluid
	float3* pos = (float3*)(pFluidBuf +__mul24(Idx,solver.bytestride));

	int hx = pos->x / solver.gridSize;
	int hy = pos->y /solver.gridSize;
	int hz = pos->z /solver.gridSize;
    
	if(Idx >= numParts)
	{
		pHashBuff[Idx] = NULL_HASH;
		pHashValueBuff[Idx] = Idx;
	}
	else
	{
		uint key = hashFunc(hx,hy,hz);
		//Save to the hash Buffer
	  	pHashBuff[Idx] = key;
		pHashValueBuff[Idx] = Idx;
	}

	__syncthreads();

};




__global__ void SortParticles(char* pFluidBuf,char** pSortFluidBuf, int numParts,uint* pHashBuff,uint* pHashValueBuff, int2* gridBuffer)
{
	uint Idx = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	extern __shared__ uint sharedHash[];
	uint hash;
	if(Idx < numParts)
	{
		//Stagger the sorted hashes onto the shared memory
		hash = pHashBuff[Idx];
		sharedHash[threadIdx.x+1] = hash;
		    if (Idx > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = pHashBuff[Idx-1];
        }
	}

	__syncthreads();

	if(Idx < numParts)
	{
		//If its the first particle or the hash doesn't correspond to the memory below it
		if(Idx == 0 || hash!=sharedHash[threadIdx.x])
		{
			//Then we have found the start of the hash
			gridBuffer[hash].x = Idx;
			if(Idx > 0)
			{
				//Therefore it is also the end of the previous hash
				gridBuffer[sharedHash[threadIdx.x]].y = Idx;
			}
		}
		if(Idx == numParts-1)
		{
			//If its the last particle then it is the last hash
			gridBuffer[hash].y = Idx+1;
		}


		//Get the pointer to the source
		char* src = pFluidBuf + pHashValueBuff[Idx]*solver.bytestride;

		//Reset values
		*(float3*)(src+FORCE_STRIDE) = make_float3(0,0,0);
		*(float*)(src+DENS_STRIDE) = 0.0f;
		*(float*)(src+PRESS_STRIDE) = 0.0f;

		//Store the pointer into the sorted buffer
		pSortFluidBuf[Idx] = src;

		/*
		//Begin sorting the particle
		char* src = pFluidBuf + pHashValueBuff[Idx]*solver.bytestride;
		char* dest = pSortFluidBuf + Idx*solver.bytestride;

		//Position
		*(float3*)dest = *(float3*)src;
		//Velocity
		*(float3*)(dest+VEL_STRIDE) = *(float3*)(src+VEL_STRIDE);
		*(float3*)(dest+VEL_EVAL_STRIDE) = *(float3*)(src+VEL_EVAL_STRIDE);
		*(float3*)(dest+FORCE_STRIDE) = make_float3(0,0,0);
		*(float*)(dest+MASS_STRIDE) = *(float*)(src+MASS_STRIDE);
		*(float*)(dest+DENS_STRIDE) = 0.0f;
		*(float*)(dest+VIS_STRIDE) = *(float*)(src+VIS_STRIDE);
		*(float*)(dest+PRESS_STRIDE) = 0.0f;
		*(float*)(dest+GAS_STRIDE) = *(float*)(src+GAS_STRIDE);
		*(float*)(dest+REST_STRIDE) = *(float*)(src+REST_STRIDE);
		*/
		
	}
	__syncthreads();

}

__device__ float ContributeDensity(int Idx,float3* iPos,uint hashKey,char** pSortFluidBuf,int2* gridBuffer)
{
	//uint hashKey = hashFunc(gridPos.x,gridPos.y,gridPos.z);
	//Load the start of the hash
	int min = gridBuffer[hashKey].x;
	if(min == -1) //If it's -1 then there are no particles and we leave
		return 0.0f;

	int max = gridBuffer[hashKey].y;
	float3* jPos;
	float mass,density;
	float dSq;
	float h2;
	float c,difx,dify,difz;
	density = 0.0f;
	//int particlesFound=0;
	for(int i = min; i< max; i++)
	{
		if(i == Idx)
			continue;
		//Get the position
		jPos = 	(float3*)(pSortFluidBuf[i]);
		mass = *(float*)(pSortFluidBuf[i] + MASS_STRIDE);
		//Calculate the distance squared
		difx = iPos->x - jPos->x;
		dify = iPos->y - jPos->y;
		difz = iPos->z - jPos->z;
		dSq = difx*difx + dify*dify + difz*difz;
		h2 = (solver.smoothradius)*(solver.smoothradius);
		if(h2 > dSq)
		{
			c = (h2 - dSq);

			density += mass*c*c*c*solver.kernal; //contribute density

		}
	}



	return density;
};


__device__ int3 CalcGridPos(float3 pos)
{
	int3 gridPos;
	gridPos.x = floor(pos.x / solver.gridSize);
	gridPos.y = floor(pos.y / solver.gridSize);
	gridPos.z = floor(pos.z / solver.gridSize);
	return gridPos;
};


__global__ void ComputeDensityAndPressure(char** pSortFluidBuf,int numParts,int2* gridBuffer)
{
	//calculate particle index
	uint Idx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
	float sum = 0.0f;
	if(Idx < numParts)
	{
		char* particle = pSortFluidBuf[Idx];
		//get the particle data for density and the gas constant
		float3* pos = (float3*)particle;
		float restDens = *(float*)(particle + REST_STRIDE);
		float gasConst = *(float*)(particle + GAS_STRIDE);

		*(float*)(particle + PRESS_STRIDE) = 0.0f;

		//Calculate the max and mins
		///float3 radmax = *pos + solver.smoothradius;
		//float3 radmin = *pos - solver.smoothradius;
		//radmax /= solver.gridSize;
		//radmin /= solver.gridSize;

		/*
		gridMin.x = (int)radmin.x;
		gridMin.y = (int)radmin.y;
		gridMin.z = (int)radmin.z;

		gridMax.x = (int)radmax.x;
		gridMax.y = (int)radmax.y;
		gridMax.z = (int)radmax.z;
		
		int3 gridPos = CalcGridPos(*pos);
		*/

		//int gridCount = 0;

		int3 gridMin = CalcGridPos(*pos - solver.smoothradius);
		int3 gridMax = CalcGridPos(*pos + solver.smoothradius);	

		//Now perform a search
		for(int z = gridMin.z; z <= gridMax.z; z++)
		{
			for(int y = gridMin.y; y<=gridMax.y; y++)
			{
				for(int x = gridMin.x; x<=gridMax.x; x++)
				{
					//int3 nPos = gridPos + make_int3(x,y,z);
					uint hash = hashFunc(x,y,z);
						//gridCount++;
						//Contribute density
						sum += ContributeDensity(Idx,pos,hash,pSortFluidBuf,gridBuffer);
					
				}
			}
		}

		//gridCount = gridCount;
        //Compute the final density and pressure
		*(float*)(particle + DENS_STRIDE) = sum;
		*(float*)(particle + PRESS_STRIDE) = (sum - restDens)*(gasConst);
		
		

	}




}



__device__ void ContributeForces(float3* forces,int Idx,char* iPart,uint hashKey,char** pSortFluidBuf,int2* gridBuffer)
{
	int min = gridBuffer[hashKey].x;

	if(min == -1) //If it's -1 then there are no particles and we leave
		return;


	int max = gridBuffer[hashKey].y;
	char* jPart;
	float rNorm;

	float3 r;
	float3 u;
	float fFactor;
	float c,jDens,iDens,jMass,iPress,jPress;

	//Get iParticle values
	iDens = *(float*)(iPart + DENS_STRIDE);
	iPress = *(float*)(iPart + PRESS_STRIDE);

	for(int i = min; i< max; i++)
	{
		if(i == Idx)
			continue;

		jPart = pSortFluidBuf[i];
		jPress = *(float*)(jPart + PRESS_STRIDE);
		jDens = *(float*)(jPart + DENS_STRIDE);
		jMass = *(float*)(jPart + MASS_STRIDE);

		if(jDens == 0.0f)
			continue;


		


		
		//Calculate r
		r = *(float3*)iPart - *(float3*)jPart;

		u = *(float3*)(jPart + VEL_STRIDE)- *(float3*)(iPart + VEL_STRIDE);
		//calculate |r|
		rNorm = length(r);

		if(rNorm > solver.smoothradius)
			continue;

		c = (solver.smoothradius - rNorm);



		/*
					pTerm = -part->Density*fluidj->Mass*pressureKernal*c*c/norm;
			float firstPress = part->Pressure/(part->Density*part->Density);
			float secPress = fluidj->Pressure/(fluidj->Density*fluidj->Density);
			float finalPress = firstPress + secPress;
			pTerm *=finalPress;
			*/

		
		
        fFactor = -iDens*jMass*solver.pressKernal*c*c/rNorm;
        fFactor *= ((iPress)/(iDens*iDens)) + ((jPress)/(jDens*jDens));

		forces->x += r.x*fFactor;
		forces->y += r.y*fFactor;
		forces->z += r.z*fFactor;
		//Apply viscosity
		fFactor = (*(float*)(iPart + VIS_STRIDE))*jMass*solver.viscKernal*c/(jDens);
		forces->x += u.x*fFactor;
		forces->y += u.y*fFactor;
		forces->z += u.z*fFactor;

		


	}
};

__global__ void ComputeForces(char** pSortFluidBuf,int numParts,int2* gridBuffer)
{
	//calculate particle index
	uint Idx = blockIdx.x*blockDim.x + threadIdx.x;
	float sum = 0.0f;
	float* force;
	if(Idx < numParts)
	{


		char* particle = pSortFluidBuf[Idx];
		*(float3*)(particle + FORCE_STRIDE) = make_float3(0,0,0);
		float3* pos = (float3*)particle;

		float3* force = (float3*)(particle + FORCE_STRIDE);

		int3 gridMin = CalcGridPos(*pos - solver.smoothradius);
		int3 gridMax = CalcGridPos(*pos + solver.smoothradius);	

		//Now perform a search
		for(int z = gridMin.z; z <= gridMax.z; z++)
		{
			for(int y = gridMin.y; y<=gridMax.y; y++)
			{
				for(int x = gridMin.x; x<=gridMax.x; x++)
				{
					uint hashKey = hashFunc(x,y,z);
						ContributeForces(force,Idx,particle,hashKey,pSortFluidBuf,gridBuffer);
				}
			}
		}
	}
}

__global__ void AdvanceEuler(float dt, char** pSortFluidBuf,int numParts)
{
	float3* vel;
		float3* pos;
		float3* force;
		float3 accel;
	uint Idx = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(Idx < numParts)
	{
		char* particle = pSortFluidBuf[Idx];
		vel = (float3*)(particle + VEL_STRIDE);
		force = (float3*)(particle + FORCE_STRIDE);
		pos = (float3*)(particle + POS_STRIDE);
	   //Now integrate the position
	   pos->x += vel->x*dt;
	   pos->y += vel->y*dt;
	   pos->z += vel->z*dt;

	   	if(*(float*)(particle + DENS_STRIDE) == 0.0f)
			accel = make_float3(0,0,0);

		accel.z += -9.81f;

		float speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
		if(speed > solver.maxSpeed*solver.maxSpeed)
		{
			accel *= solver.maxSpeed/sqrt(speed);
		}
	   //advance velocity
	   //Calculate velocity
	   accel = *force;
	   accel /= *(float*)(particle + DENS_STRIDE);
	   accel *= dt;
	   //add it to the particles velocity
	   vel->x += accel.x;
	   vel->y += accel.y;
	   vel->z += accel.z;

	   	   //Z BOUNDARY WITH OPEN TOP
	   if(pos->z < -solver.simSize.z-EPSILON)
	   {
		   pos->z = -solver.simSize.z;
		   *vel *= solver.velDamp;
	   }

	   ////////////X BOUNDARIES/////////////////
	   if(pos->x < -solver.simSize.x-EPSILON)
	   {
		   pos->x = -solver.simSize.x;
		   *vel *= solver.velDamp;
	   }
	   if(pos->x > solver.simSize.x+EPSILON)
	   {
		   pos->x = solver.simSize.x;
		   *vel *= solver.velDamp;
	   }
	   ///////////Y BOUNDARIES//////////////////
	   	   if(pos->y < -solver.simSize.y-EPSILON)
	   {
		   pos->y = -solver.simSize.y;
		   *vel *= solver.velDamp;
	   }
	   if(pos->y > solver.simSize.y+EPSILON)
	   {
		   pos->y = solver.simSize.y;
		   *vel *= solver.velDamp;
	   }

	   

	  
	}
	 __syncthreads ();

};

__global__ void AdvanceLeap(float dt, char** pSortFluidBuf,int numParts)
{
	float3* vel;
	float3* velEval;
	float3* pos;
	float3* force;
	float3 accel;
	uint Idx = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if(Idx < numParts)
	{
		char* particle = pSortFluidBuf[Idx];
		vel = (float3*)(particle + VEL_STRIDE);
		velEval = (float3*)(particle + VEL_EVAL_STRIDE);
		force = (float3*)(particle + FORCE_STRIDE);
		pos = (float3*)(particle + POS_STRIDE);


		accel = *force;

	   	if(*(float*)(particle + DENS_STRIDE) == 0.0f)
			accel = make_float3(0,0,0);
		else
			accel /= *(float*)(particle + DENS_STRIDE);



		//Add gravity
		accel.z -= 9.81f;

	   //Speed limiting
		float speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
		if(speed > solver.maxSpeed*solver.maxSpeed)
		{
			accel *= solver.maxSpeed/sqrt(speed);
		}

	   
	   accel *= dt;
	   //
	   accel += *velEval;

	   //Now integrate the position
	   *pos += accel*dt;


	   //Limit boundary

	   /*

	   		if(particles[i]->Position->z < -1-EPSILON)
		{
			particles[i]->Position->z = -1;
			vNext *= -0.1f;
		}
		
		if(particles[i]->Position->x < -boxSize-EPSILON)
		{
			particles[i]->Position->x = -boxSize;
			vNext *= -0.1f;
		}
		else if(particles[i]->Position->x > boxSize+EPSILON)
		{
			particles[i]->Position->x = boxSize;
			vNext *= -0.1f;
		}
		if(particles[i]->Position->y < -boxSize-EPSILON)
		{
			particles[i]->Position->y = -boxSize;
			vNext *= -0.1f;
		}
		else if(particles[i]->Position->y > boxSize+EPSILON)
		{
			particles[i]->Position->y = boxSize;
			vNext *= -0.1f;
		}
		*/
	   //Z BOUNDARY WITH OPEN TOP
	   if(pos->z < -solver.simSize.z-EPSILON)
	   {
		   pos->z = -solver.simSize.z;
		   accel *= solver.velDamp;
	   }

	   ////////////X BOUNDARIES/////////////////
	   if(pos->x < -solver.simSize.x-EPSILON)
	   {
		   pos->x = -solver.simSize.x;
		   accel *= solver.velDamp;
	   }
	   if(pos->x > solver.simSize.x+EPSILON)
	   {
		   pos->x = solver.simSize.x;
		   accel *= solver.velDamp;
	   }
	   ///////////Y BOUNDARIES//////////////////
	   	   if(pos->y < -solver.simSize.y-EPSILON)
	   {
		   pos->y = -solver.simSize.y;
		   accel *= solver.velDamp;
	   }
	   if(pos->y > solver.simSize.y+EPSILON)
	   {
		   pos->y = solver.simSize.y;
		   accel *= solver.velDamp;
	   }


	   //Calculate th midpoint to find the velocity (for viscosity)
	   *vel = accel + *velEval;
	   *vel *= 0.5f;
	   
	   *velEval = accel;

	   /*
	   		//Evaluate the velocity
		part->Velocity->x = vNext.x + part->VelEval->x;
		part->Velocity->y = vNext.y + part->VelEval->y;
		part->Velocity->z = vNext.z + part->VelEval->z;
		*part->Velocity *= 0.5f;
		part->VelEval->Set(vNext.x,vNext.y,vNext.z);
		*/
	  
	}
	 __syncthreads ();

};




