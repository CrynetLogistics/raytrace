#pragma once
#include "Mesh.h"
#include "Math.h"

class Sphere: public Mesh
{
private:
	vertex_t centre;
	colour_t colour;
	float radius;
	Material material;
public:
	__device__ vertex_t getCentre(void);
	__device__ float getRadius(void);
	__device__ float getIntersectionParameter(vector_t lightRay) override;
	__device__ float getShadowedStatus(vector_t lightRay, float t, Light light) override;
	__device__ vector_t getNormal(vertex_t pos, vector_t incoming) override;
	__device__ Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour, Material material);
	__device__ colour_t getColour(void) override;
	__device__ Material getMaterial(void) override;
	__device__ ~Sphere(void);
};

