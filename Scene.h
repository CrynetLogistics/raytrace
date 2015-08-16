#pragma once
#include "Objects/Camera.h"
#include "Objects/Light.h"
#include "Objects/Sphere.h"
#include "Objects/Plane.h"
#include "stdlib.h"
#include "stdio.h"

class Scene
{
private:
	Camera camera;
	Light light;
	Mesh **meshes;
	int numOfMeshes;
public:
	Scene(void);
	Camera getCamera(void);
	Mesh* getMesh(int number);
	Light getLight(void);
	void addSphere(float centreX, float centreY, float centreZ, float radius, colour_t col, Material material);
	void addPlane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, Material material);
	//THIS IS SETLIGHT FOR NOW - CAN ONLY HANDLE 1 LIGHT
	void addLight(float posX, float posY, float posZ, float intensity);
	int getNumOfMeshes(void);
	~Scene(void);
};

