#include "Camera.h"


__host__ __device__ Camera::Camera(void)
{
	central_direction.x0 = 0;
	central_direction.y0 = 0;
	central_direction.z0 = 0;

	central_direction.xt = 0;
	central_direction.yt = 3;
	central_direction.zt = 0;

	gridSize = (float)0.01;
}

__host__ __device__ float Camera::getGridSize(void){
	return gridSize;
}

__host__ __device__ vector_t Camera::getLocDir(void){
	return central_direction;
}

__host__ __device__ Camera::~Camera(void)
{
}
