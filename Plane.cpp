#include "Plane.h"

//TODO:HANDLE PARALLELOGRAMS AND NO PLANAR SURFACES
//TODO:INFER VERTEX4 FROM V1,2,3 AND SQUARE PROPERTY

Plane::Plane(void){
	colour_t col2;
	col2.r = 255;
	col2.g = 77;
	col2.b = 99;

	vertex_t v1;
	vertex_t v2;
	vertex_t v3;
	vertex_t v4;
	v1.x = 1;v1.y = 8;v1.z = -1;
	v2.x = 8;v2.y = 8;v2.z = -1;
	v3.x = 1;v3.y = 15;v3.z = 3;
	v4.x = 8;v4.y = 15;v4.z = 3;

	this->v1 = v1;
	this->v2 = v2;
	this->v3 = v3;
	this->v4 = v4;

	this->colour = colour;
	vector_t side1(v1,v2);
	vector_t side2(v1,v3);
	//vector_t side3(v4,v2);
	//vector_t side4(v4,v3);

	vector_t normal = side1.directionCrossProduct(side2);
	a = normal.xt;
	b = normal.yt;
	c = normal.zt;
	d = normal.directionDotProduct(vector_t(0,0,0,v1.x,v1.y,v1.z));
	this->normal = normal;
}

Plane::Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour)
{
	this->v1 = v1;
	this->v2 = v2;
	this->v3 = v3;
	this->v4 = v4;

	this->colour = colour;
	vector_t side1(v1,v2);
	vector_t side2(v1,v3);
	//vector_t side3(v4,v2);
	//vector_t side4(v4,v3);

	vector_t normal = side1.directionCrossProduct(side2);
	a = normal.xt;
	b = normal.yt;
	c = normal.zt;
	d = normal.directionDotProduct(vector_t(0,0,0,v1.x,v1.y,v1.z));
	this->normal = normal;
}

float Plane::getIntersectionParameter(vector_t lightRay, Light light){
	float t;
	vector_t lightSource(0,0,0,lightRay.x0,lightRay.y0,lightRay.z0);
	vector_t lightDirection(0,0,0,lightRay.xt,lightRay.yt,lightRay.zt);
	t = (d - normal.directionDotProduct(lightSource))/normal.directionDotProduct(lightDirection);

	vector_t side1(v1,v2);
	vector_t side2(v1,v3);
	vector_t side3(v4,v2);
	vector_t side4(v4,v3);

	vertex_t planePoint = lightRay.getPosAtParameter(t);
	vector_t pointFromV1(v1, planePoint);
	vector_t pointFromV4(v4, planePoint);

	if(side1.directionDotProduct(pointFromV1) > 0 &&
	   side2.directionDotProduct(pointFromV1) > 0 &&
	   side3.directionDotProduct(pointFromV4) > 0 &&
	   side4.directionDotProduct(pointFromV4) > 0){
		return t;
	}else{
		return 0;
	}
}

colour_t Plane::getColour(void){
	return colour;
}

Plane::~Plane(void)
{
}
