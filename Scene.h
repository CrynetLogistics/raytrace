#pragma once
#include "support.h"
#include "BinTree/BinTree.cuh"

class Scene
{
private:
	Camera camera;
	Light light;
	Mesh **meshes;
	int numOfMeshes;
	int totalMeshes;
	uint32_t* textureData;
	colour_t horizonColour;
	BinTree* partitioningHierachy;
public:
	__host__ __device__ Scene(int totalMeshes, uint32_t* textureData);
	__host__ __device__ Camera getCamera(void);
	__host__ __device__ Mesh* getMesh(int number);
	__host__ __device__ Light getLight(void);
	__host__ __device__ void setHorizonColour(colour_t horizonColour);
	__host__ __device__ colour_t getHorizonColour(void);
	__host__ __device__ void addSphere(float centreX, float centreY, float centreZ, float radius, colour_t col, materialType_t material);
	__host__ __device__ void addPlane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, materialType_t material);
	__host__ __device__ void addTri(vertex_t v1, vertex_t v2, vertex_t v3, colour_t colour, materialType_t material);
	//THIS IS SETLIGHT FOR NOW - CAN ONLY HANDLE 1 LIGHT
	__host__ __device__ void addLight(float posX, float posY, float posZ, float intensity);
	__host__ __device__ int getNumOfMeshes(void);
	__host__ __device__ uint32_t* getTexture(void);
	__host__ __device__ Mesh** getMeshes(void);
	__host__ __device__ void buildBSPBVH(void);
	__host__ __device__ float collisionDetect(vector_t ray, Mesh** mesh);
	__host__ __device__ ~Scene(void);
};

