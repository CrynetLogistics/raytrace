#include "Plane.h"

__device__ Plane::Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, Material material)
	:material(material){
	this->v1 = v1;
	this->v2 = v2;
	this->v3 = v3;
	this->v4 = v4;

	this->colour = colour;
	vector_t side1(v1,v2);
	vector_t side2(v1,v3);

	vector_t normal = side1.directionCrossProduct(side2);
	a = normal.xt;
	b = normal.yt;
	c = normal.zt;
	d = normal.directionDotProduct(vector_t(0,0,0,v1.x,v1.y,v1.z));
	this->normal = normal;
	this->material = material;
}

__device__ float Plane::getIntersectionParameter(vector_t lightRay){
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

//for self shadowing only (isShadowed)
__device__ float Plane::getShadowedStatus(vector_t lightRay, float t, Light light){
	vertex_t pos = lightRay.getPosAtParameter(t);
	vector_t lightVector(pos.x, pos.y, pos.z, light.getPos().x-pos.x, light.getPos().y-pos.y, light.getPos().z-pos.z);
	vector_t cameraVector(pos.x, pos.y, pos.z, -1*lightRay.xt, -1*lightRay.yt, -1*lightRay.zt);
	float index = normal.directionDotProduct(cameraVector)*normal.directionDotProduct(lightVector);
	if(index>0){
		return -1;
	}else{
		return 1;
	}
}

//TODO:CURRENTLY UNUSED FEATURE - NORMAL RETURNED IN THE SAME DIRECTION AS REFLECTED RAY
__device__ vector_t Plane::getNormal(vertex_t pos, vector_t incoming){
	//if(normal.directionDotProduct(incoming)>0){
	//	normal.xt = -1*normal.xt;
	//	normal.yt = -1*normal.yt;
	//	normal.zt = -1*normal.zt;
	//}
	normal.x0 = pos.x;
	normal.y0 = pos.y;
	normal.z0 = pos.z;
	return normal;
}

__device__ colour_t Plane::getColour(void){
	return colour;
}

__device__ Material Plane::getMaterial(void){
	return material;
}

__device__ Plane::~Plane(void){
}
