#pragma once
#include "../Scene.h"
//#include "../Auxiliary/extremum_t.cuh"
class BinTreeNode
{
private:
	extremum_t extremum;

	BinTreeNode* leftChild;
	BinTreeNode* rightChild;

	Mesh** meshes;
	int numOfMeshes;

public:
	__host__ __device__ BinTreeNode(Mesh** meshes, int numOfMeshes);
	__host__ __device__ void propagateTree(void);
	__host__ __device__ ~BinTreeNode(void);
};
