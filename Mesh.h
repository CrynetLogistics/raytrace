#pragma once
#include "Light.h"
#include "Auxiliary/vector_t.h"
#include "Material.h"

class Mesh
{
public:
	__device__ Mesh(void);
	__device__ virtual float getIntersectionParameter(vector_t lightRay) = 0;
	__device__ virtual colour_t getColour(void) = 0;
	__device__ virtual float getShadowedStatus(vector_t lightRay, float t, Light light) = 0;
	__device__ virtual vector_t getNormal(vertex_t pos, vector_t incoming) = 0;
	__device__ virtual Material getMaterial(void) = 0;
	__device__ ~Mesh(void);
};
