#pragma once
#include "Camera.h"
#include "Light.h"
#include "Sphere.h"
#include "Plane.h"
#include "Tri.h"
#include "stdlib.h"
#include "stdio.h"

class Scene
{
private:
	Camera camera;
	Light light;
	Mesh **meshes;
	int numOfMeshes;
	int totalMeshes;
public:
	__device__ Scene(int totalMeshes);
	__host__ __device__ Camera getCamera(void);
	__host__ __device__ Mesh* getMesh(int number);
	__host__ __device__ Light getLight(void);
	__device__ void addSphere(float centreX, float centreY, float centreZ, float radius, colour_t col, Material material);
	__device__ void addPlane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, Material material);
	__device__ void addTri(vertex_t v1, vertex_t v2, vertex_t v3, colour_t colour, Material material);
	//THIS IS SETLIGHT FOR NOW - CAN ONLY HANDLE 1 LIGHT
	__host__ __device__ void addLight(float posX, float posY, float posZ, float intensity);
	__host__ __device__ int getNumOfMeshes(void);
	__device__ ~Scene(void);
};

