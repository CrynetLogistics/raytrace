#pragma once
#include "structures.h"
#include "math.h"
class vector_t
{
public:
	//structure variables
	float x0;
	float y0;
	float z0;
	float xt;
	float yt;
	float zt;
	//..............................
	__host__ __device__ vector_t(void);
	__host__ __device__ vector_t(float x0, float y0, float z0, float xt, float yt, float zt);
	__host__ __device__ vector_t(vertex_t origin, vertex_t destination);
	__host__ __device__ float calculateDistance(float t);
	__host__ __device__ vertex_t getPosAtParameter(float t);
	__host__ __device__ float directionDotProduct(vector_t dotterand);
	__host__ __device__ vector_t directionCrossProduct(vector_t crosserand);
	__host__ __device__ float directionMagnitude(void);
	__host__ __device__ ~vector_t(void);
};

