#pragma once
#include "BinTreeNode.cuh"
#include "../Queue/Queue.h"

//The buildBSPBVH signal is propagated through the end of the kernel initScene function
//this travels here via the Scene class which feeds the Binary tree with mesh data
class BinTree
{
private:
	BinTreeNode* root;
	int maxTreeHeight;
public:
	__host__ __device__ BinTree(Mesh** meshes, int numOfMeshes);
	__host__ __device__ void buildTree(int maxTreeHeight);

	//The mesh is returned through the pointer and the distance
	//is returned by the function
	__host__ __device__ float findCollisionMesh(vector_t ray, Mesh** mesh);

	__host__ __device__ ~BinTree(void);
};
