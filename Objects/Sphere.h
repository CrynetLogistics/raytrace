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
	__host__ __device__ vertex_t getCentre(void);
	__host__ __device__ float getRadius(void);
	__host__ __device__ float getIntersectionParameter(vector_t lightRay) override;
	__host__ __device__ float getShadowedStatus(vector_t lightRay, float t, Light light) override;
	__host__ __device__ vector_t getNormal(vertex_t pos, vector_t incoming) override;
	__host__ __device__ Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour, Material material);
	__host__ __device__ colour_t getColour(void) override;
	__host__ __device__ Material getMaterial(void) override;
	__host__ __device__ ~Sphere(void);
};

