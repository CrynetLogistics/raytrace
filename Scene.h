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
	__host__ Scene(void);
	__host__ __device__ Camera getCamera(void);
	__host__ __device__ Mesh* getMesh(int number);
	__host__ __device__ Light getLight(void);
	__host__ void addSphere(float centreX, float centreY, float centreZ, float radius, colour_t col, Material material);
	__host__ void addPlane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, Material material);
	__host__ __device__ //THIS IS SETLIGHT FOR NOW - CAN ONLY HANDLE 1 LIGHT
	__host__ __device__ void addLight(float posX, float posY, float posZ, float intensity);
	__host__ __device__ int getNumOfMeshes(void);
	__host__ ~Scene(void);
};

