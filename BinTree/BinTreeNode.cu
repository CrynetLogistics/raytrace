#include "BinTreeNode.cuh"

__host__ __device__ BinTreeNode::BinTreeNode(Mesh** meshes, int numOfMeshes){
	this->meshes = meshes;
	this->numOfMeshes = numOfMeshes;

	extremum_t init = meshes[0]->findExtremum();
	extremum = extremum_t(&init);
	for(int i=1; i<numOfMeshes; i++){
		extremum.mergeExtrema(meshes[i]->findExtremum());
	}
}

__host__ __device__ void BinTreeNode::propagateTree(void){

	leftChild->propagateTree();
	rightChild->propagateTree();
}

__host__ __device__ BinTreeNode::~BinTreeNode(void){
}
