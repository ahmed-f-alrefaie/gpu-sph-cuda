#include "stdafx.h"
#include "SPHSolver.h"
#include "SPHSolverHost.cuh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

#define PI 3.141512
#define EPSILON 0.0001
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
		//copy vel
		memcpy(mPartBuffer+partPtrIdx+VEL_STRIDE,&mParticles[i].vel.x,sizeof(float)); // x
		memcpy(mPartBuffer+partPtrIdx+VEL_STRIDE+4,&mParticles[i].vel.y,sizeof(float)); // y
		memcpy(mPartBuffer+partPtrIdx+VEL_STRIDE+8,&mParticles[i].vel.z,sizeof(float)); // z
				//copy vel
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

	for(int i =0; i<mTotalParticles; i++)
	{
		Fluid* f_i = &mParticles[i];
		f_i->density = 0.0f;
	
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

				float c = h2 - r2;
				f_i->density += f_i->mass*defaultKernal*c*c*c;
			}
		}
		f_i->press = (f_i->density - f_i->restdens)*f_i->gasconst;

	}
}

void SPHSolver::ComputeForcesBrute()
{


	float norm,c,pTerm,vTerm;
	Vector3D r;
	Vector3D u;
	float pressTotal = 0;

	for(int i =0; i<mTotalParticles; i++)
	{
		Fluid* f_i = &mParticles[i];
		f_i->force.x = 0;
		f_i->force.y = 0;
		f_i->force.z = 0;
	
		if(f_i->density == 0)
			continue;

		for(int j=0; j < mTotalParticles; j++)
		{
			Fluid* f_j = &mParticles[j];
			if(f_i == f_j)
				continue;
		 if(f_j->density == 0)
			continue;


			r.x = f_i->pos.x - f_j->pos.x;
			r.y = f_i->pos.y - f_j->pos.y;
			r.z = f_i->pos.z - f_j->pos.z;

			u.x = f_j->vel.x - f_i->vel.x;
			u.y = f_j->vel.y - f_i->vel.y;
			u.z = f_j->vel.z - f_i->vel.z;
			norm = r.Distance();

			if(norm > mSmoothRadius)
				continue;

			c = (mSmoothRadius- norm);

			pTerm = -f_i->density*f_j->mass*pressureKernal*c*c/norm;

			pressTotal = (f_i->press/(f_i->density*f_i->density)) +
			(f_j->press/(f_j->density*f_j->density));
			pTerm *=pressTotal;

			
			f_i->force.x += r.x*pTerm;
		    f_i->force.y += r.y*pTerm;
			f_i->force.z += r.z*pTerm;

			

			float vTerm = f_i->visc*f_j->mass*viscosityKernal*c/f_j->density;
			Vector3D viscosityForce = u*vTerm;
			f_i->force += viscosityForce;
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
	//ComputeForcesBrute();
	//Advance by euler
	AdvanceSPH(dt);
	//AdvanceLeapFrog(dt);
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
	//	glColor3f(1.0f,0.0f,0.0f);
		//glVertex3f(mParticles[i].pos.x*2,mParticles[i].pos.y*2,mParticles[i].pos.z*2);
	}
	glEnd();


}

void SPHSolver::AdvanceLeapFrog(float dt)
{
	for(int i = 0; i < mTotalParticles; i++)
	{
		Fluid* f_i = &mParticles[i];



		Vector3D accel = f_i->force;
		if(f_i->density == 0.0f)
			accel.Set(0,0,0);
		else
		    accel *= 1/f_i->density;
		
        accel.z += -9.81f;
		//accel += *part->ArtifVis;

		
		float speed = accel.DistanceSquared();
		if(speed > 200*200)
		{
			accel *= 200/std::sqrt(speed);
		}

        
		//Calculate the new veloctiy
		Vector3D vNext = accel; 
		vNext *= dt;

		vNext += f_i->velEval;
		
		//vNext += *part->VelocityAdjust;

		f_i->pos.x += vNext.x*dt;
		f_i->pos.y += vNext.y*dt;
		f_i->pos.z += vNext.z*dt;

		float boxSize = 1.0f;


		if(f_i->pos.z < -1-EPSILON)
		{
			f_i->pos.z = -1;
			vNext.z *= -0.01f;
		}
		
		if(f_i->pos.x < -boxSize-EPSILON)
		{
			f_i->pos.x = -boxSize;
			vNext.x *= -0.01f;
		}
		else if(f_i->pos.x > boxSize+EPSILON)
		{
			f_i->pos.x = boxSize;
			vNext.x *= -0.01f;
		}
		if(f_i->pos.y < -boxSize-EPSILON)
		{
			f_i->pos.y = -boxSize;
			vNext.y *= -0.01f;
		}
		else if(f_i->pos.y > boxSize+EPSILON)
		{
			f_i->pos.y = boxSize;
			vNext.y *= -0.01f;
		}
		
		//Evaluate the velocity
		f_i->vel.x = vNext.x + f_i->velEval.x;
		f_i->vel.y = vNext.y + f_i->velEval.y;
		f_i->vel.z = vNext.z + f_i->velEval.z;
		f_i->vel *= 0.5f;
		f_i->velEval.Set(vNext.x,vNext.y,vNext.z);



	}
}