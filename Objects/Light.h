#pragma once
#include "../Auxiliary/structures.h"
class Light
{
private:
	vertex_t pos;
	float intensity;
public:
	Light(float posX, float posY, float posZ, float intensity);
	float getIntensity(void);
	vertex_t getPos(void);
	~Light(void);
};
