#include "Sphere.h"
#include "math.h"
#include "Light.h"
#include "vector_t.h"

Sphere::Sphere(void)
{
	centre.x = 0;
	centre.y = 10;
	centre.z = 0;
	radius = 2;
	colour.r = 255;
	colour.g = 0;
	colour.b = 255;
}

Sphere::Sphere(float centreX, float centreY, float centreZ, float radius, colour_t colour){
	centre.x = centreX;
	centre.y = centreY;
	centre.z = centreZ;
	this->radius = radius;
	this->colour.r = colour.r;
	this->colour.g = colour.g;
	this->colour.b = colour.b;
}

float Sphere::getRadius(void){
	return radius;
}

vertex_t Sphere::getCentre(void){
	return centre;
}

float Sphere::getIntersectionParameter(vector_t lightRay, Light light){
	float acx = lightRay.x0-centre.x;
	float acy = lightRay.y0-centre.y;
	float acz = lightRay.z0-centre.z;
	float a = lightRay.xt*lightRay.xt+lightRay.yt*lightRay.yt+lightRay.zt*lightRay.zt;
	float b = 2*(lightRay.xt*acx + lightRay.yt*acy + lightRay.zt*acz);
	float c = acx*acx + acy*acy + acz*acz - radius*radius;
	
	//std::cout<<sqrt(b*b-4*a*c);
	if(b*b-4*a*c>=0){
		float t = (-b-sqrt(b*b-4*a*c))/(2*a);
		//std::cout<<t/1000;
		//return t*60;
		return t;//lightRay.calculateDistance(t);
	}else{
		return 0;
	}
}

bool Sphere::getShadowedStatus(vector_t lightRay, float t, Light light){
	vertex_t pos = lightRay.getPosAtParameter(t);
	vector_t normalVector(pos.x, pos.y, pos.z, pos.x-centre.x, pos.y-centre.y, pos.z-centre.z);
	vector_t lightVector(pos.x, pos.y, pos.z, light.getPos().x-pos.x, light.getPos().y-pos.y, light.getPos().z-pos.z);
	vector_t cameraVector(pos.x, pos.y, pos.z, -1*lightRay.xt, -1*lightRay.yt, -1*lightRay.zt);
	float index = normalVector.directionDotProduct(cameraVector)*normalVector.directionDotProduct(lightVector);
	if(index>0){
		return false;
	}else{
		return true;
	}
}

colour_t Sphere::getColour(){
	return colour;
}

Sphere::~Sphere(void)
{
}
