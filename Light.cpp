#include "Light.h"


Light::Light(void)
{
	pos.x = 0;
	pos.y = 5;
	pos.z = 8;
	intensity = 10;
}

void Light::setParams(float posX, float posY, float posZ, float intensity){
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
