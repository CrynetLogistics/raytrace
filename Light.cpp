#include "Light.h"


Light::Light(float posX, float posY, float posZ, float intensity){
	pos.x = posX;
	pos.y = posY;
	pos.z = posZ;
	this->intensity = intensity;
}

float Light::getIntensity(void){
	return intensity;
}

vertex_t Light::getPos(void){
	return pos;
}

Light::~Light(void)
{
}
