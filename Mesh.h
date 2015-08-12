#pragma once
#include "structures.h"
#include "vector_t.h"
#include "Light.h"

class Mesh
{
public:
	Mesh(void);
	virtual float getIntersectionParameter(vector_t lightRay, Light light) = 0;
	virtual colour_t getColour(void) = 0;
	virtual bool getShadowedStatus(vector_t lightRay, float t, Light light) = 0;
	virtual vector_t getNormal(vertex_t pos) = 0;
	~Mesh(void);
};
