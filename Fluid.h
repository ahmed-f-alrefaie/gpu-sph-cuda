#include "stdafx.h"
#include "Vector3D.h"

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

struct Fluid
{
	//Position, velocity and forces
	Vector3D pos,vel,force,velEval;
	//Attributes of the particle
	float mass,density,visc,press,gasconst,restdens;
};