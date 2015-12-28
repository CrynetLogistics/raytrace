#include "Camera.h"


__host__ __device__ Camera::Camera(void){
	central_direction.x0 = 0;
	central_direction.y0 = 0;
	central_direction.z0 = 0;

	central_direction.xt = 0;
	central_direction.yt = 3;
	central_direction.zt = 0;

	gridSize = (float)0.01;

	locDir = vector_t(0,0,0,0,3,0);

	ZOOM_FACTOR = 0.01f;
}

__host__ __device__ float Camera::getGridSize(void){
	return gridSize;
}

__host__ __device__ vector_t Camera::getLocDir(void){
	return central_direction;
}

__host__ __device__ Camera::~Camera(void){
}

__host__ __device__ vector_t Camera::getThisLocationDirection(int i, int j, int SCREEN_WIDTH, int SCREEN_HEIGHT){
	vector_t thisLocDir = vector_t();
	thisLocDir.x0 = locDir.x0;
	thisLocDir.y0 = locDir.y0;
	thisLocDir.z0 = locDir.z0;

	thisLocDir.xt = locDir.xt + (float)(i-SCREEN_WIDTH/2)*ZOOM_FACTOR;
	thisLocDir.yt = locDir.yt;
	thisLocDir.zt = locDir.zt + (float)(SCREEN_HEIGHT/2-j)*ZOOM_FACTOR;

	return thisLocDir;
}