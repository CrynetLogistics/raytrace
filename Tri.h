#pragma once
#include "Mesh.h"

class Tri: public Mesh
{
private:
	vertex_t v1;
	vertex_t v2;
	vertex_t v3;
	colour_t colour;
	Material material;
	//FOR EQUATION: ax+by+cz=d
	vector_t normal;
	float a;
	float b;
	float c;
	float d;
public:
	__device__ Tri(vertex_t v1, vertex_t v2, vertex_t v3, colour_t colour, materialType_t materialType);
	__device__ ~Tri(void);
	__device__ float getIntersectionParameter(vector_t lightRay) override;
	__device__ colour_t getColour(vertex_t position) override;
	__device__ float getShadowedStatus(vector_t lightRay, float t, Light light) override;
	__device__ vector_t getNormal(vertex_t pos, vector_t incoming) override;
	__device__ Material getMaterial(void) override;
};

