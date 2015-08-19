#pragma once
#include "Mesh.h"

//v1 and v4 provided are on opposite sides of a diagonal
class Plane: public Mesh
{
private:
	vertex_t v1;
	vertex_t v2;
	vertex_t v3;
	vertex_t v4;
	colour_t colour;
	Material material;
	//FOR EQUATION: ax+by+cz=d
	vector_t normal;
	float a;
	float b;
	float c;
	float d;
public:
	__device__ Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, Material material);
	__device__ ~Plane(void);
	__device__ float getIntersectionParameter(vector_t lightRay) override;
	__device__ colour_t getColour(void) override;
	__device__ float getShadowedStatus(vector_t lightRay, float t, Light light) override;
	__device__ vector_t getNormal(vertex_t pos, vector_t incoming) override;
	__device__ Material getMaterial(void) override;

//public:
//	bool getShadowedStatus(vector_t lightRay, float t, Light light);
//	colour_t getColour();
};

