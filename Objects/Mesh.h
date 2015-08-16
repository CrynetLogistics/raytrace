#pragma once
#include "Light.h"
#include "../Auxiliary/vector_t.h"
#include "../Auxiliary/Material.h"

class Mesh
{
public:
	Mesh(void);
	virtual float getIntersectionParameter(vector_t lightRay, Light light) = 0;
	virtual colour_t getColour(void) = 0;
	virtual float getShadowedStatus(vector_t lightRay, float t, Light light) = 0;
	virtual vector_t getNormal(vertex_t pos, vector_t incoming) = 0;
	virtual Material getMaterial(void) = 0;
	~Mesh(void);
};
