#pragma once
#include "Auxiliary/vector_t.h"
class Camera
{
private:
	vector_t central_direction;
	vector_t locDir;
	float gridSize;
	float ZOOM_FACTOR;
public:
	__host__ __device__ float getGridSize(void);
	__host__ __device__ vector_t getLocDir(void);
	__host__ __device__ Camera(void);
	__host__ __device__ ~Camera(void);
	__host__ __device__ vector_t getThisLocationDirection(int i, int j, int SCREEN_WIDTH, int SCREEN_HEIGHT);
	__host__ __device__ vector_t getThisLocationDirection(int i, int j, int SCREEN_WIDTH, int SCREEN_HEIGHT, int MSAA_SAMPLES, int MSAA_INDEX);
};

