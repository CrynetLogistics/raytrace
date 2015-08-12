#pragma once
#include "structures.h"
#include "vector_t.h"
#include "Light.h"
#include "Mesh.h"
class Sphere: public Mesh
{
private:
	vertex_t centre;
	colour_t colour;
	float radius;
public:
	vertex_t getCentre(void);
	float getRadius(void);
	float getIntersectionParameter(vector_t lightRay, Light light) override;
	bool getShadowedStatus(vector_t lightRay, float t, Light light) override;
	vector_t getNormal(vertex_t pos) override;
	Sphere(void);
	Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour);
	colour_t getColour(void) override;
	~Sphere(void);
};

