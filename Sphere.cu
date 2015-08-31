#include "Sphere.h"

#define PI 3.14159f
#define SPHERICAL_MAP 1

__device__ Sphere::Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour, materialType_t materialType){
	material.initMaterial(materialType);
	centre.x = centreX;
	centre.y = centreY;
	centre.z = centreZ;
	this->radius = radius;
	this->colour.r = colour.r;
	this->colour.g = colour.g;
	this->colour.b = colour.b;
}

__device__ Sphere::Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour, uint32_t* textureData){
	material.initMaterial(textureData);
	centre.x = centreX;
	centre.y = centreY;
	centre.z = centreZ;
	this->radius = radius;
	this->colour.r = colour.r;
	this->colour.g = colour.g;
	this->colour.b = colour.b;
}

__device__ float Sphere::getRadius(void){
	return radius;
}

__device__ vertex_t Sphere::getCentre(void){
	return centre;
}

__device__ float Sphere::getIntersectionParameter(vector_t lightRay){
	float acx = lightRay.x0-centre.x;
	float acy = lightRay.y0-centre.y;
	float acz = lightRay.z0-centre.z;
	float a = lightRay.xt*lightRay.xt+lightRay.yt*lightRay.yt+lightRay.zt*lightRay.zt;
	float b = 2*(lightRay.xt*acx + lightRay.yt*acy + lightRay.zt*acz);
	float c = acx*acx + acy*acy + acz*acz - radius*radius;
	
	if(b*b-4*a*c>=0){
		float t = (-b-sqrt(b*b-4*a*c))/(2*a);
		return t;
	}else{
		return 0;
	}
}

//returns a negative for false and positive for true
//magnitude of the number from 0 to 1 to indicate distance from extremum
__device__ float Sphere::getShadowedStatus(vector_t lightRay, float t, Light light){
	vertex_t pos = lightRay.getPosAtParameter(t);
	vector_t normalVector(pos.x, pos.y, pos.z, pos.x-centre.x, pos.y-centre.y, pos.z-centre.z);
	vector_t lightVector(pos.x, pos.y, pos.z, light.getPos().x-pos.x, light.getPos().y-pos.y, light.getPos().z-pos.z);
	vector_t cameraVector(pos.x, pos.y, pos.z, -1*lightRay.xt, -1*lightRay.yt, -1*lightRay.zt);
	float index = normalVector.directionDotProduct(cameraVector)*normalVector.directionDotProduct(lightVector);
	float extremumProjection = normalVector.directionDotProduct(lightVector)/(normalVector.directionMagnitude()*lightVector.directionMagnitude());
	if(index>0){
		return -1 * abs(extremumProjection);// false;
	}else{
		return abs(extremumProjection);//true;
	}
}

__device__ vector_t Sphere::getNormal(vertex_t pos, vector_t incoming){
	vector_t normalVector(pos.x, pos.y, pos.z, pos.x-centre.x, pos.y-centre.y, pos.z-centre.z);

	return normalVector;
}

__device__ colour_t Sphere::getColour(vertex_t position){
	if(material.isTextured()>=2){
		vector_t r(centre, position);
		vector_t s(centre, position);
		s.zt = 0;

		vector_t k_unit(0,0,0,0,0,1);
		vector_t i_unit(0,0,0,1,0,0);

		float cosine_phi = k_unit.directionDotProduct(r)/r.directionMagnitude();
		float cosine_theta = i_unit.directionDotProduct(s)/s.directionMagnitude();

		int x,y;
		if(SPHERICAL_MAP){
			if(s.yt>0){//theta < pi
				x = 300*(1-acosf(cosine_theta)/PI);
			}else{//theta >= pi
				x = 300*acosf(cosine_theta)/PI;
			}
			y = 300*(acosf(cosine_phi))/PI;
		}else{
			x = 600*(cosine_theta+1)/2;
			y = 300*(cosine_phi+1)/2;
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

__device__ Material Sphere::getMaterial(void){
	return material;
}

__device__ Sphere::~Sphere(void){
}
