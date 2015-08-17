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
	Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, Material material);
	~Plane(void);
	float getIntersectionParameter(vector_t lightRay) override;
	colour_t getColour(void) override;
	float getShadowedStatus(vector_t lightRay, float t, Light light) override;
	vector_t getNormal(vertex_t pos, vector_t incoming) override;
	Material getMaterial(void) override;

//public:
//	bool getShadowedStatus(vector_t lightRay, float t, Light light);
//	colour_t getColour();
};

