#include "Light.h"


__host__ __device__ Light::Light(float posX, float posY, float posZ, float intensity){
	pos.x = posX;
	pos.y = posY;
	pos.z = posZ;
	this->intensity = intensity;
}

__host__ __device__ float Light::getIntensity(void){
	return intensity;
}

__host__ __device__ vertex_t Light::getPos(void){
	return pos;
}

__host__ __device__ Light::~Light(void)
{
}
