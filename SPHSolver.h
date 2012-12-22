#include <vector_types.h>
#include "Fluid.h"
#include <vector>
#include <GL\freeglut.h>



class SPHSolver
{
private:
	//char* mPartBuffer; //The data used to transfer into CUDA
	std::vector<Fluid> mParticles; 	//A dynamic list of the particle data
	int mTotalParticles;
	float mSmoothRadius;

	//Kernals
    float defaultKernal;
	float divDefaultKernal;
	float laplacian;
	float pressureKernal;
	float viscosityKernal;


	void GenerateCUDABuffer();
	char* mPartBuffer;
public:

	SPHSolver();
	SPHSolver(float pSmoothRadius);
	~SPHSolver();
	void AddNewParticle(float x,float y,float z, float mass,float vis,float gasconst,float rest);
	void SetupCUDASolver();
	void ComputeKernals();
	void ComputeDensityBrute();
	void ComputeForcesBrute();
	void AdvanceLeapFrog(float dt);
	inline int GetParticleCount(){return mTotalParticles;}
	void Advance(float dt);
	void Draw(float* viewMat);

};