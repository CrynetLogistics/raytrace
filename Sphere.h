#pragma once
#include "Mesh.h"
#include "math.h"

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
	__host__ __device__ Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour, materialType_t materialType);
	__host__ __device__ Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour, uint32_t* textureData);
	__host__ __device__ colour_t getColour(vertex_t position) override;
	__host__ __device__ Material getMaterial(void) override;
	__host__ __device__ int isContainedWithin(extremum_t ex) override;
	__host__ __device__ extremum_t findExtremum(void) override;
	__host__ __device__ ~Sphere(void);
};

