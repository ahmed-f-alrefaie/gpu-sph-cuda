#include "stdafx.h"
#include "SPHSolver.h"
#include "SPHSolverHost.cuh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

#define PI 3.141512

SPHSolver::SPHSolver()
{
	mSmoothRadius = 0.0456f;
	mTotalParticles = 0;
	ComputeKernals();
}

SPHSolver::SPHSolver(float pSmoothRadius)
{
	mSmoothRadius = pSmoothRadius;
	mTotalParticles = 0;
	ComputeKernals();
}


void SPHSolver::AddNewParticle(float x,float y,float z, float mass,float vis,float gasconst,float rest)
{
	//Add a new particle to the dynamic list
	mParticles.push_back(Fluid());
	int idx = mParticles.size()-1;
	Fluid* fldptr = &mParticles[idx];

	fldptr->pos.Set(x,y,z);
	fldptr->vel.Set(0,0,0);
	fldptr->force.Set(0,0,0);
	fldptr->mass = mass;
	fldptr->gasconst = gasconst;
	fldptr->press = 0;
	fldptr->density = 0;
	fldptr->visc = vis;
	fldptr->restdens = rest;
	mTotalParticles = mParticles.size();

}

void SPHSolver::GenerateCUDABuffer(){

	if(mTotalParticles ==0)
	{
		std::cout<<"No fluid particles to put into buffer"<<std::endl;
		return;
	}
	size_t fluidsize = sizeof(Fluid);
	int partPtrIdx;
	mPartBuffer = (char*)malloc(mTotalParticles* fluidsize);
	for(int i = 0; i < mTotalParticles; i++)
	{
		partPtrIdx = i*fluidsize;
		
		//Copy position
		memcpy(mPartBuffer+partPtrIdx+POS_STRIDE,&mParticles[i].pos.x,sizeof(float)); // x
		memcpy(mPartBuffer+partPtrIdx+POS_STRIDE+4,&mParticles[i].pos.y,sizeof(float)); // y
		memcpy(mPartBuffer+partPtrIdx+POS_STRIDE+8,&mParticles[i].pos.z,sizeof(float)); // z
		//copy velocity
		memcpy(mPartBuffer+partPtrIdx+VEL_STRIDE,&mParticles[i].vel.x,sizeof(float)); // x
		memcpy(mPartBuffer+partPtrIdx+VEL_STRIDE+4,&mParticles[i].vel.y,sizeof(float)); // y
		memcpy(mPartBuffer+partPtrIdx+VEL_STRIDE+8,&mParticles[i].vel.z,sizeof(float)); // z
				//copy velocity
		memcpy(mPartBuffer+partPtrIdx+VEL_EVAL_STRIDE,&mParticles[i].vel.x,sizeof(float)); // x
		memcpy(mPartBuffer+partPtrIdx+VEL_EVAL_STRIDE+4,&mParticles[i].vel.y,sizeof(float)); // y
		memcpy(mPartBuffer+partPtrIdx+VEL_EVAL_STRIDE+8,&mParticles[i].vel.z,sizeof(float)); // z
		//copy Force 
		memcpy(mPartBuffer+partPtrIdx+FORCE_STRIDE,&mParticles[i].force.x,sizeof(float)); // x
		memcpy(mPartBuffer+partPtrIdx+FORCE_STRIDE+4,&mParticles[i].force.y,sizeof(float)); // y
		memcpy(mPartBuffer+partPtrIdx+FORCE_STRIDE+8,&mParticles[i].force.z,sizeof(float)); // z
		//copy mass
		memcpy(mPartBuffer+partPtrIdx+MASS_STRIDE,&mParticles[i].mass,sizeof(float));
		//copy density
		memcpy(mPartBuffer+partPtrIdx+DENS_STRIDE,&mParticles[i].density,sizeof(float));
		//copy viscosity
		memcpy(mPartBuffer+partPtrIdx+VIS_STRIDE,&mParticles[i].visc,sizeof(float));
		//copy pressure
		memcpy(mPartBuffer+partPtrIdx+PRESS_STRIDE,&mParticles[i].press,sizeof(float));
		//copy gasConstant
		memcpy(mPartBuffer+partPtrIdx+GAS_STRIDE,&mParticles[i].gasconst,sizeof(float));
		//copy rest density
		memcpy(mPartBuffer+partPtrIdx+REST_STRIDE,&mParticles[i].restdens,sizeof(float));
	}
	/*
		int i;
	for(i = 0; i < mTotalParticles; i++)
	{
		float3* pos = (float3*)(mPartBuffer + i*fluidsize);
		float* dens = (float*)(mPartBuffer + i*fluidsize + DENS_STRIDE);
		float3* force = (float3*)(mPartBuffer + i*fluidsize + FORCE_STRIDE);
		float mass = *(float*)(mPartBuffer + i*fluidsize + MASS_STRIDE);
		printf("Particle %i - x: %f y: %f z: %f density: %f fx: %f fy: %f fz: %f mass: %f\n",i,pos->x,pos->y,pos->z,*dens,force->x,force->y,force->z,mass);
	}
	*/
}

SPHSolver::~SPHSolver()
{

	//free the particle buffer
	free(mPartBuffer);
	mPartBuffer = NULL;
	//Clear the particle list
	mParticles.clear();
	ClearCUDA();
	
}

void SPHSolver::SetupCUDASolver()
{
	//Generate the CUDA buffer
	GenerateCUDABuffer();
	//Setup an CUDA stuff
	SetupCUDA(mTotalParticles,sizeof(Fluid));
    SetupSolver(mSmoothRadius, mTotalParticles);
	//Transfer to CUDA
	TransferToCUDA(mPartBuffer,mTotalParticles,sizeof(Fluid));

}


void SPHSolver::ComputeKernals()
{
	defaultKernal = 315/(64*PI*std::powf(mSmoothRadius,9));
	divDefaultKernal = -945/(32*PI*std::powf(mSmoothRadius,9));
	laplacian = -945/(32*PI*std::powf(mSmoothRadius,9));
	pressureKernal = -45/(PI*std::powf(mSmoothRadius,6));
	viscosityKernal = 45/(PI*std::powf(mSmoothRadius,6));
}
void SPHSolver::ComputeDensityBrute()
{
	int particlesFound = 0;
	for(int i =0; i<mTotalParticles; i++)
	{
		Fluid* f_i = &mParticles[i];
		f_i->density = 0.0f;
		particlesFound = 0;
		for(int j=0; j < mTotalParticles; j++)
		{
			Fluid* f_j = &mParticles[j];
			if(f_i == f_j)
				continue;

			Vector3D r = f_i->pos - f_j->pos;
			float r2 = r.DistanceSquared();
			float h2 = mSmoothRadius*mSmoothRadius;
			if(h2 > r2)
			{
				particlesFound++;
				float c = h2 - r2;
				f_i->density += f_i->mass*defaultKernal*c*c*c;
			}
		}

	}



}

void SPHSolver::Advance(float dt)
{


	//TransferToCUDA(mPartBuffer,mTotalParticles,sizeof(Fluid));
	//Insert the particles
	InsertParticles();
//	CheckHashes();

	//Compute Forces
	ComputeDensityCUDA();
	//Computer BruteDenisty
	//ComputeDensityBrute();
	ComputeForcesCUDA();

	//Advance by euler
	AdvanceSPH(dt);
//	CheckSortedParticles();

	//CheckMemory();
	//SwapParticleBuffers();
	//TransferFromCUDA(mPartBuffer,mTotalParticles,sizeof(Fluid));
}

void SPHSolver::Draw(float* viewMat)
{
	//Simple drawing mechanism
	//Transfer from cuda
	TransferFromCUDA(mPartBuffer,mTotalParticles,sizeof(Fluid));
//	TestParticleBuffer(mPartBuffer,mTotalParticles);
	//Go through and draw the positions
	glBegin(GL_POINTS);
	for(int i = 0; i < mTotalParticles; i++)
	{

		float3* position = (float3*)(mPartBuffer + i*sizeof(Fluid));
		glColor3f(1.0f,1.0f,1.0f);
		glVertex3f(position->x*2,position->y*2,position->z*2);
	}
	glEnd();


}
	