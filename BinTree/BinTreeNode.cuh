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

	int maxChildrenHeight;

	//MAKE OBSOLETE
	//repetitionIndex = number of times the box has had the
	//					same number of meshes
	int repetitionIndex;
public:
	__host__ __device__ BinTreeNode(Mesh** meshes, int numOfMeshes, extremum_t extremum, int repetitionIndex, int maxChildrenHeight);
	__host__ __device__ BinTreeNode(Mesh** meshes, int numOfMeshes, int maxChildrenHeight);
	__host__ void propagateTree();
	__device__ void propagateTree(Stack<BinTreeNode*> *d_unPropagatedNodes);
	__host__ __device__ bool containsRay(vector_t ray);
	__host__ __device__ bool isLeaf(void);
	__host__ __device__ BinTreeNode* getLeftChild(void);
	__host__ __device__ BinTreeNode* getRightChild(void);
	__host__ __device__ int getNumOfMeshes(void);
	__host__ __device__ Mesh* getMeshAt(int index);
	__host__ __device__ ~BinTreeNode(void);
};
