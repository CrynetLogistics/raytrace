#pragma once
#include "Auxiliary/structures.h"
class Light
{
private:
	vertex_t pos;
	float intensity;
public:
	__host__ __device__ Light(float posX, float posY, float posZ, float intensity);
	__host__ __device__ float getIntensity(void);
	__host__ __device__ vertex_t getPos(void);
	__host__ __device__ ~Light(void);
};
