#pragma once
#include "structures.h"
#include "vector_t.h"
#include "Light.h"

//v1 and v4 provided are on opposite sides of a diagonal
class Plane
{
public:
	Plane(void);
	Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour);
	~Plane(void);
	float getIntersectionParameter(vector_t lightRay, Light light);
	colour_t getColour(void);
private:
	vertex_t v1;
	vertex_t v2;
	vertex_t v3;
	vertex_t v4;
	colour_t colour;
	//FOR EQUATION: ax+by+cz=d
	vector_t normal;
	float a;
	float b;
	float c;
	float d;
//public:
//	bool getShadowedStatus(vector_t lightRay, float t, Light light);
//	colour_t getColour();
};

