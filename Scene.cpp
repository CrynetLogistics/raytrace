#include "Scene.h"

//default lamp
Scene::Scene(void)
	:light(0, 0, 10, 10)
{
	numOfSpheres = 0;
	spheres = (Sphere*)calloc(1, sizeof(Sphere));
	plane = (Plane*)calloc(1, sizeof(Plane));
	numOfMeshes = 0;
	meshes = (Mesh**)calloc(1, sizeof(Mesh*));
}

Scene::Scene(float lightX, float lightY, float lightZ, float intensity)
	:light(lightX, lightY, lightZ, intensity)
{
	numOfSpheres = 0;
	spheres = (Sphere*)calloc(1, sizeof(Sphere));
}

Mesh* Scene::getMesh(int number){
	if(number<=numOfMeshes){
		return meshes[number];
	}else{
		printf("ERROR: THAT MESH DOES NOT EXSIST, DEFAULT MESH RETURNED");
		return *(meshes+0);
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

	numOfMeshes++;
	meshes = (Mesh**)realloc(meshes, numOfMeshes*sizeof(Mesh*));

	Sphere *t = new Sphere(centreX, centreY, centreZ, radius, col);

	Mesh *m = t;
	meshes[numOfMeshes-1] = m;
}

void Scene::addPlane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour){
	numOfMeshes++;
	meshes = (Mesh**)realloc(meshes, numOfMeshes*sizeof(Mesh*));

	Plane *p = new Plane(v1, v2, v3, v4, colour);

	Mesh *m = p;
	meshes[numOfMeshes-1] = m;
}

void Scene::addLight(float posX, float posY, float posZ, float intensity){
	light = Light(posX, posY, posZ, intensity);
}

int Scene::getNumOfMeshes(void){
	return numOfMeshes;
}

Light Scene::getLight(void){
	return light;
}

Scene::~Scene(void)
{
	free(meshes);
}
