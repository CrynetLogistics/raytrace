#include "BinTree.cuh"

__host__ __device__ BinTree::BinTree(Scene* scene){
	root = new BinTreeNode(scene->getMeshes, scene->getNumOfMeshes);
}

__host__ __device__ void BinTree::buildTree(void){
	root->propagateTree();
}

__host__ __device__ BinTree::~BinTree(void){
	delete(root);
}
