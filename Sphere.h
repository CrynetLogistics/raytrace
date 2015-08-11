#pragma once
#include "structures.h"
#include "vector_t.h"
#include "Light.h"
class Sphere
{
private:
	vertex_t centre;
	colour_t colour;
	float radius;
public:
	vertex_t getCentre(void);
	float getRadius(void);
	float getIntersectionParameter(vector_t lightRay, Light light);
	bool getShadowedStatus(vector_t lightRay, float t, Light light);
	Sphere(void);
	Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour);
	colour_t getColour();
	~Sphere(void);
};

