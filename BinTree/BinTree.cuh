#pragma once
#include "BinTreeNode.cuh"
class BinTree
{
private:
	BinTreeNode* root;
public:
	__host__ __device__ BinTree(Scene* scene);
	__host__ __device__ void buildTree(void);
	__host__ __device__ ~BinTree(void);
};
