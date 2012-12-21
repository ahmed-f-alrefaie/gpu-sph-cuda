#include <math.h>
#pragma once
	//Quick and dirty vector class
	class Vector3D
	{
	public:
		float x,y,z;

		inline Vector3D(float x=0, float y=0, float z=0){ this->x = x; this->y = y; this->z = z;}

		inline float Distance(){float dist = DistanceSquared(); if(dist != 0.0){ return sqrtf(dist);} else return 0;}
			inline float DistanceSquared(){ return (x*x + y*y + z*z);}
			void FastNormalize();
		void Normalize();
		inline void Set(float x, float y, float z){this->x = x; this->y = y; this->z = z;};
		//Operator overloads
		Vector3D & operator=(const Vector3D& val);
		Vector3D & operator+(const Vector3D& val); 
		Vector3D & operator+=(const Vector3D& rhs);
		Vector3D & operator-=(const Vector3D& rhs);
		Vector3D & Vector3D::operator-(const Vector3D &rhs);
		Vector3D & operator*(const Vector3D& val);
		Vector3D & operator*(const float& val);
		Vector3D & Vector3D::operator*=(const float& val);
	

	};

	inline Vector3D & Vector3D::operator+=(const Vector3D &rhs)
{
	this->x += rhs.x;
	this->y += rhs.y;
	this->z += rhs.z;
	return *this;
}

	inline Vector3D & Vector3D::operator-=(const Vector3D &rhs)
{
	this->x -= rhs.x;
	this->y -= rhs.y;
	this->z -= rhs.z;
	return *this;
}
inline Vector3D & Vector3D::operator-(const Vector3D &rhs)
{
	return Vector3D(this->x - rhs.x,this->y-rhs.y,this->z - rhs.z);
}
inline Vector3D & Vector3D::operator+(const Vector3D &rhs)
{
	return Vector3D(this->x + rhs.x,this->y+rhs.y,this->z + rhs.z);
}

inline Vector3D & Vector3D::operator*(const float& val)
{
	return Vector3D(this->x*val,
	this->y*val,
	this->z*val);
}
inline Vector3D & Vector3D::operator*=(const float& val)
{
	this->x*=val;
	this->y*=val;
	this->z*=val;
	return *this;
}


