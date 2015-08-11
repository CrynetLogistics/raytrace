#include "Camera.h"


Camera::Camera(void)
{
	central_direction.x0 = 0;
	central_direction.y0 = 0;
	central_direction.z0 = 0;

	central_direction.xt = 0;
	central_direction.yt = 3;
	central_direction.zt = 0;

	gridSize = 0.01;
}

float Camera::getGridSize(void){
	return gridSize;
}

vector_t Camera::getLocDir(void){
	return central_direction;
}

Camera::~Camera(void)
{
}
