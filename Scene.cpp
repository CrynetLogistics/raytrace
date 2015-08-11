#include "Scene.h"
#include "Camera.h"
#include "Light.h"
#include "Sphere.h"
#include "stdlib.h"
#include "stdio.h"

Scene::Scene(void)
{
	numOfSpheres = 0;
	spheres = (Sphere*)calloc(1, sizeof(Sphere));
	plane = (Plane*)calloc(1, sizeof(Plane));
	*plane = Plane();
}

Scene::Scene(float lightX, float lightY, float lightZ, float intensity){
	numOfSpheres = 0;
	spheres = (Sphere*)calloc(1, sizeof(Sphere));
	light.setParams(lightX, lightY, lightZ, intensity);
}

Sphere Scene::getSphere(int number){
	if(number<=numOfSpheres){
		return spheres[number];
	}else{
		printf("ERROR: THAT SHPERE DOES NOT EXSIST, DEFAULT SPHERE RETURNED");
		return spheres[0];
	}
}

Camera Scene::getCamera(void){
	return camera;
}

Plane Scene::getPlane(int number){
	number++;
	return *plane;
}

void Scene::addSphere(float centreX, float centreY, float centreZ, float radius, colour_t col){
	numOfSpheres++;
	spheres = (Sphere*)realloc(spheres, numOfSpheres*sizeof(Sphere));
	
	//Sphere newSphere(centreX, centreY, centreZ, radius);
	//spheres[numOfSpheres-1] = newSphere;
	//EQUALS THIS:
	spheres[numOfSpheres-1] = Sphere(centreX, centreY, centreZ, radius, col);
}

int Scene::getNumOfSpheres(void){
	return numOfSpheres;
}

Light Scene::getLight(void){
	return light;
}

Scene::~Scene(void)
{
	free(spheres);
}
