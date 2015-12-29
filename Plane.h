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
	__host__ __device__ Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, materialType_t materialType);
	__host__ __device__ Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, uint32_t* textureData);
	__host__ __device__ ~Plane(void);
	__host__ __device__ float getIntersectionParameter(vector_t lightRay) override;
	__host__ __device__ colour_t getColour(vertex_t position) override;
	__host__ __device__ float getShadowedStatus(vector_t lightRay, float t, Light light) override;
	__host__ __device__ vector_t getNormal(vertex_t pos, vector_t incoming) override;
	__host__ __device__ Material getMaterial(void) override;
	__host__ __device__ int isContainedWithin(vertex_t extremum1, vertex_t extremum2) override;
	__host__ __device__ extremum_t findExtremum(void) override;
};

