#pragma once
#include "Camera.h"
#include "Light.h"
#include "Sphere.h"
#include "Plane.h"
#include "stdlib.h"
#include "stdio.h"

class Scene
{
private:
	Plane *plane;
	int numOfSpheres;
	Camera camera;
	Light light;
	Sphere *spheres;
	Mesh **meshes;
	int numOfMeshes;
public:
	Scene(void);
	Scene(float lightX, float lightY, float lightZ, float intensity);
	Camera getCamera(void);
	Mesh* getMesh(int number);
	Plane getPlane(int number);
	Light getLight(void);
	void addSphere(float centreX, float centreY, float centreZ, float radius, colour_t col);
	void addPlane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour);
	//THIS IS SETLIGHT FOR NOW - CAN ONLY HANDLE 1 LIGHT
	void addLight(float posX, float posY, float posZ, float intensity);
	int getNumOfMeshes(void);
	~Scene(void);
};

