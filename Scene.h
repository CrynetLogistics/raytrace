#include "Camera.h"
#include "Light.h"
#include "Sphere.h"
#include "Plane.h"
#pragma once

class Scene
{
private:
	Plane *plane;
	int numOfSpheres;
	Camera camera;
	Light light;
	Sphere *spheres;
public:
	Scene(void);
	Scene(float lightX, float lightY, float lightZ, float intensity);
	Camera getCamera(void);
	Sphere getSphere(int number);
	Plane getPlane(int number);
	Light getLight(void);
	void addSphere(float centreX, float centreY, float centreZ, float radius, colour_t col);
	int getNumOfSpheres(void);
	~Scene(void);
};

