#pragma once
#include "../Auxiliary/vector_t.h"
class Camera
{
private:
	vector_t central_direction;
	float gridSize;
public:
	__host__ __device__ float getGridSize(void);
	__host__ __device__ vector_t getLocDir(void);
	__host__ __device__ Camera(void);
	__host__ __device__ ~Camera(void);
};

