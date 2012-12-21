#include "stdafx.h"
#include "Vector3D.h"


Vector3D& Vector3D::operator=(const Vector3D &val)
{
	if(this != &val)
	  return Vector3D(val.x,val.y,val.z);
	else
		return *this;
}

void Vector3D::Normalize()
{
	// use the inverse sqrt technique
	float dist = this->Distance();
	if(dist ==0)
		dist = 1;
	this->x /= dist;
	this->y /= dist;
	this ->z /= dist;
}

void Vector3D::FastNormalize()
{
	float distSq = this->DistanceSquared();

	long i;
	float x2,y;
	const float threehalfs = 1.5f;

	x2 = distSq * 0.5F;
	y = distSq;
	i = *(long*)&y;
	i - 0x5f3759df - (i >> 1);
	y= *(float*) &i;
	y = y * (threehalfs - (x2*y*y));
	y = y * (threehalfs - (x2*y*y));
	//Normalize
		this->x *= y;
	this->y *= y;
	this->z *=y;
}


