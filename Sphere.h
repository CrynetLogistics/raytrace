#pragma once
#include "Mesh.h"
#include "Math.h"

class Sphere: public Mesh
{
private:
	vertex_t centre;
	colour_t colour;
	float radius;
	float reflectivity;
public:
	vertex_t getCentre(void);
	float getRadius(void);
	float getIntersectionParameter(vector_t lightRay, Light light) override;
	float getShadowedStatus(vector_t lightRay, float t, Light light) override;
	vector_t getNormal(vertex_t pos, vector_t incoming) override;
	float getReflectivity(void) override;
	Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour, float reflectivity);
	colour_t getColour(void) override;
	~Sphere(void);
};

