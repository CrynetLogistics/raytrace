#include "vector_t.h"

__host__ __device__ vector_t::vector_t(void)
{
}

__host__ __device__ vector_t::vector_t(vertex_t origin, vertex_t destination){
	x0 = 0;
	y0 = 0;
	z0 = 0;
	xt = destination.x - origin.x;
	yt = destination.y - origin.y;
	zt = destination.z - origin.z;
}

__host__ __device__ vector_t::vector_t(float x0, float y0, float z0, float xt, float yt, float zt)
{
	this->x0 = x0;
	this->y0 = y0;
	this->z0 = z0;
	this->xt = xt;
	this->yt = yt;
	this->zt = zt;
}

__host__ __device__ float vector_t::calculateDistance(float t){
	float xdist = xt*t;
	float ydist = yt*t;
	float zdist = zt*t;
	return sqrt(xdist*xdist+ydist*ydist+zdist*zdist);
}

__host__ __device__ vertex_t vector_t::getPosAtParameter(float t){
	vertex_t v;
	v.x = x0 + xt*t;
	v.y = y0 + yt*t;
	v.z = z0 + zt*t;
	return v;
}

__host__ __device__ float vector_t::directionDotProduct(vector_t dotterand){
	return xt*dotterand.xt + yt*dotterand.yt + zt*dotterand.zt;
}

__host__ __device__ vector_t vector_t::directionCrossProduct(vector_t crosserand){
	vector_t vector;
	vector.x0 = 0;
	vector.y0 = 0;
	vector.z0 = 0;
	vector.xt = yt*crosserand.zt - zt*crosserand.yt;
	vector.yt = zt*crosserand.xt - xt*crosserand.zt;
	vector.zt = xt*crosserand.yt - yt*crosserand.xt;
	return vector;
}

__host__ __device__ float vector_t::directionMagnitude(void){
	return sqrt(xt*xt+yt*yt+zt*zt);
}

__host__ __device__ vector_t::~vector_t(void)
{
}
