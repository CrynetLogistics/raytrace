#include "Tri.h"

__device__ Tri::Tri(vertex_t v1, vertex_t v2, vertex_t v3, colour_t colour, materialType_t materialType){
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

__device__ float Tri::getIntersectionParameter(vector_t lightRay){
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
__device__ float Tri::getShadowedStatus(vector_t lightRay, float t, Light light){
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

__device__ vector_t Tri::getNormal(vertex_t pos, vector_t incoming){
	vector_t normalFromPos;
	normalFromPos.xt = normal.xt;
	normalFromPos.yt = normal.yt;
	normalFromPos.zt = normal.zt;
	normalFromPos.x0 = pos.x;
	normalFromPos.y0 = pos.y;
	normalFromPos.z0 = pos.z;

	return normalFromPos;
}

__device__ colour_t Tri::getColour(vertex_t position){
	return colour;
}

__device__ Material Tri::getMaterial(void){
	return material;
}

__device__ Tri::~Tri(void){
}