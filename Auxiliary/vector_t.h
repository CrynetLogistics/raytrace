#pragma once
#include "structures.h"
#include "math.h"
class vector_t
{
public:
	//REMEMBER TO MAKE THESE PRIVATE
	float x0;
	float y0;
	float z0;
	float xt;
	float yt;
	float zt;
	//..............................
	vector_t(void);
	vector_t(float x0, float y0, float z0, float xt, float yt, float zt);
	vector_t(vertex_t origin, vertex_t destination);
	float calculateDistance(float t);
	vertex_t getPosAtParameter(float t);
	float directionDotProduct(vector_t dotterand);
	vector_t directionCrossProduct(vector_t crosserand);
	float directionMagnitude(void);
	~vector_t(void);
};

