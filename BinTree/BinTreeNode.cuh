#pragma once
#include "../support.h"
#include "../Auxiliary/extremum_t.cuh"
#include "../Auxiliary/structures.h"

class BinTreeNode
{
private:
	bool hasChildren;
	extremum_t extremum;

	BinTreeNode* leftChild;
	BinTreeNode* rightChild;

	Mesh** meshes;
	int numOfMeshes;
	int INITIAL_MESHES;

	//repetitionIndex = number of times the box has had the
	//					same number of meshes
	int repetitionIndex;
public:
	__host__ __device__ BinTreeNode(Mesh** meshes, int numOfMeshes, extremum_t extremum, int repetitionIndex);
	__host__ __device__ BinTreeNode(Mesh** meshes, int numOfMeshes);
	__host__ __device__ void propagateTree(int maxTreeHeight);
	__host__ __device__ bool containsRay(vector_t ray);
	__host__ __device__ bool isLeaf(void);
	__host__ __device__ BinTreeNode* getLeftChild(void);
	__host__ __device__ BinTreeNode* getRightChild(void);
	__host__ __device__ int getNumOfMeshes(void);
	__host__ __device__ Mesh* getMeshAt(int index);
	__host__ __device__ ~BinTreeNode(void);
};
