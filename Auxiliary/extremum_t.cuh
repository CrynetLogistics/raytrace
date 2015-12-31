#pragma once
#include "structures.h"

class extremum_t
{
private:
	float lowX;
	float lowY;
	float lowZ;

	float highX;
	float highY;
	float highZ;
	
	__host__ __device__ float max3(float a, float b, float c);
public:
	__host__ __device__ extremum_t(void);
	__host__ __device__ extremum_t(extremum_t* ex);
	__host__ __device__ extremum_t(vertex_t e);
	__host__ __device__ extremum_t(vertex_t e1, vertex_t e2);
	__host__ __device__ extremum_t(float lowX, float lowY, float lowZ, float highX, float highY, float highZ);
	__host__ __device__ void factorInExtremum(vertex_t e);
	__host__ __device__ void mergeExtrema(extremum_t ex);
	__host__ __device__ vertex_t getLowExtremum(void);
	__host__ __device__ vertex_t getHighExtremum(void);

	__host__ __device__ extremum_t getPartitionedLowExtremum(void);
	__host__ __device__ extremum_t getPartitionedHighExtremum(void);

	__host__ __device__ ~extremum_t(void);
};
