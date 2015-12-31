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
	__host__ __device__ Tri(vertex_t v1, vertex_t v2, vertex_t v3, colour_t colour, materialType_t materialType);
	__host__ __device__ ~Tri(void);
	__host__ __device__ float getIntersectionParameter(vector_t lightRay) override;
	__host__ __device__ colour_t getColour(vertex_t position) override;
	__host__ __device__ float getShadowedStatus(vector_t lightRay, float t, Light light) override;
	__host__ __device__ vector_t getNormal(vertex_t pos, vector_t incoming) override;
	__host__ __device__ Material getMaterial(void) override;
	__host__ __device__ int isContainedWithin(extremum_t ex) override;
	__host__ __device__ extremum_t findExtremum(void) override;
};

