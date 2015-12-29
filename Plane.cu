#include "Plane.h"

#define HORIZONTAL_MAP 0

__host__ __device__ Plane::Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, materialType_t materialType){
	material.initMaterial(materialType);
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

__host__ __device__ Plane::Plane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, uint32_t* textureData){
	material.initMaterial(textureData);
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

__host__ __device__ float Plane::getIntersectionParameter(vector_t lightRay){
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
__host__ __device__ float Plane::getShadowedStatus(vector_t lightRay, float t, Light light){
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

__host__ __device__ vector_t Plane::getNormal(vertex_t pos, vector_t incoming){
	vector_t normalFromPos;
	normalFromPos.xt = normal.xt;
	normalFromPos.yt = normal.yt;
	normalFromPos.zt = normal.zt;
	normalFromPos.x0 = pos.x;
	normalFromPos.y0 = pos.y;
	normalFromPos.z0 = pos.z;

	return normalFromPos;
}

__host__ __device__ colour_t Plane::getColour(vertex_t position){
	if(material.isTextured()>=2){
		float x0 = position.x;
		float x_hat = v1.x;

		float x_p = v3.x;
		float x_pp = v2.x;

		float y0 = position.y;
		float y_hat = v1.y;
		
		float y_p = v3.y;
		float y_pp = v2.y;

		int x,y;
		if(HORIZONTAL_MAP){
			if(x_p != x_hat){
				x = (300*(x0 - x_hat))/(x_p - x_hat);
			}else{
				x = (300*(y0 - y_hat))/(y_p - y_hat);
			}

			if(x_pp != x_hat){
				y = (600*(x0 - x_hat))/(x_pp - x_hat);
			}else{
				y = (600*(y0 - y_hat))/(y_pp - y_hat);
			}
		}else{
			if(x_p != x_hat){
				x = (600*(x0 - x_hat))/(x_p - x_hat);
			}else{
				x = (600*(y0 - y_hat))/(y_p - y_hat);
			}

			if(x_pp != x_hat){
				y = (300*(x0 - x_hat))/(x_pp - x_hat);
			}else{
				y = (300*(y0 - y_hat))/(y_pp - y_hat);
			}
		}

		colour_t pointColour;

		pointColour.r = (material.getTexture()[600*y+x] & 0x000000FF) >> 0;
		pointColour.g = (material.getTexture()[600*y+x] & 0x0000FF00) >> 8;
		pointColour.b = (material.getTexture()[600*y+x] & 0x00FF0000) >> 16;
		return pointColour;
	}else{
		return colour;
	}
}

__host__ __device__ int Plane::isContainedWithin(vertex_t extremum1, vertex_t extremum2){
	return 2;
}

__host__ __device__ Material Plane::getMaterial(void){
	return material;
}

__host__ __device__ extremum_t Plane::findExtremum(void){
	extremum_t e(v1, v2);
	e.factorInExtremum(v3);
	e.factorInExtremum(v4);

	return e;
}

__host__ __device__ Plane::~Plane(void){
}
