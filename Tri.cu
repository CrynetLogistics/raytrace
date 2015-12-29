#include "Tri.h"

__host__ __device__ Tri::Tri(vertex_t v1, vertex_t v2, vertex_t v3, colour_t colour, materialType_t materialType){
	material.initMaterial(materialType);
	this->v1 = v1;
	this->v2 = v2;
	this->v3 = v3;

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

__host__ __device__ float Tri::getIntersectionParameter(vector_t lightRay){
	float t;
	vector_t lightSource(0,0,0,lightRay.x0,lightRay.y0,lightRay.z0);
	vector_t lightDirection(0,0,0,lightRay.xt,lightRay.yt,lightRay.zt);
	t = (d - normal.directionDotProduct(lightSource))/normal.directionDotProduct(lightDirection);

	vector_t side1(v1,v2);
	vector_t side2(v2,v3);
	vector_t side3(v3,v1);

	vector_t normal1 = normal.directionCrossProduct(side1);
	vector_t normal2 = normal.directionCrossProduct(side2);
	vector_t normal3 = normal.directionCrossProduct(side3);

	vertex_t planePoint = lightRay.getPosAtParameter(t);
	vector_t pointFromV1(v1, planePoint);
	vector_t pointFromV2(v2, planePoint);
	vector_t pointFromV3(v3, planePoint);

	if(normal1.directionDotProduct(pointFromV1) > 0 &&
	   normal2.directionDotProduct(pointFromV2) > 0 &&
	   normal3.directionDotProduct(pointFromV3) > 0){
		return t;
	}else{
		return 0;
	}
}

//for self shadowing only (isShadowed)
__host__ __device__ float Tri::getShadowedStatus(vector_t lightRay, float t, Light light){
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

__host__ __device__ vector_t Tri::getNormal(vertex_t pos, vector_t incoming){
	vector_t normalFromPos;
	normalFromPos.xt = normal.xt;
	normalFromPos.yt = normal.yt;
	normalFromPos.zt = normal.zt;
	normalFromPos.x0 = pos.x;
	normalFromPos.y0 = pos.y;
	normalFromPos.z0 = pos.z;

	return normalFromPos;
}

__host__ __device__ colour_t Tri::getColour(vertex_t position){
	return colour;
}

__host__ __device__ Material Tri::getMaterial(void){
	return material;
}

__host__ __device__ int Tri::isContainedWithin(vertex_t extremum1, vertex_t extremum2){
	bool v1c = false;
	bool v2c = false;
	bool v3c = false;

	if((extremum1.x<v1.x && v1.x<extremum2.x) || (extremum1.x>v1.x && v1.x>extremum2.x) &&
		(extremum1.y<v1.y && v1.y<extremum2.y) || (extremum1.y>v1.y && v1.y>extremum2.y) &&
		(extremum1.z<v1.z && v1.z<extremum2.z) || (extremum1.z>v1.z && v1.z>extremum2.z)){
		v1c = true;
	}

	if((extremum1.x<v2.x && v2.x<extremum2.x) || (extremum1.x>v2.x && v2.x>extremum2.x) &&
		(extremum1.y<v2.y && v2.y<extremum2.y) || (extremum1.y>v2.y && v2.y>extremum2.y) &&
		(extremum1.z<v2.z && v2.z<extremum2.z) || (extremum1.z>v2.z && v2.z>extremum2.z)){
		v2c = true;
	}

	if((extremum1.x<v3.x && v3.x<extremum2.x) || (extremum1.x>v3.x && v3.x>extremum2.x) &&
		(extremum1.y<v3.y && v3.y<extremum2.y) || (extremum1.y>v3.y && v3.y>extremum2.y) &&
		(extremum1.z<v3.z && v3.z<extremum2.z) || (extremum1.z>v3.z && v3.z>extremum2.z)){
		v3c = true;
	}

	if(v1c&&v2c&&v3c){
		return 2;
	}else if(v1c||v2c||v3c){
		return 1;
	}else{
		return 0;
	}
}

__host__ __device__ extremum_t Tri::findExtremum(void){
	extremum_t e(v1,v2);
	e.factorInExtremum(v3);

	return e;
}

__host__ __device__ Tri::~Tri(void){
}