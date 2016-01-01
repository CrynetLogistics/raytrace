#include "BinTree.cuh"

#define EPSILON 0.001f
#define CLIPPING_DISTANCE 999

__host__ __device__ BinTree::BinTree(Mesh** meshes, int numOfMeshes){
	root = new BinTreeNode(meshes, numOfMeshes);
}

__host__ __device__ void BinTree::buildTree(int maxTreeHeight){
	this->maxTreeHeight = maxTreeHeight;
	root->propagateTree(maxTreeHeight);
}

__host__ __device__ float BinTree::findCollisionMesh(vector_t ray, Mesh** mesh){
	///NEED FAR MORE SPACE FOR THISSSS
	Stack<BinTreeNode*> collisionPotentials(maxTreeHeight);
	BinTreeNode* current = root;

	Mesh* currentBestMesh;
	float tMin = CLIPPING_DISTANCE;

	while(true){

		int totalMeshes = current->getNumOfMeshes();
		for(int i=0; i<totalMeshes; i++){

			float tCurrent = current->getMeshAt(i)->getIntersectionParameter(ray);

			if(EPSILON<tCurrent && tCurrent<tMin){// && tCurrent!=0){
				tMin = tCurrent;
				currentBestMesh = current->getMeshAt(i);
			}
		}
		
		//set up ***current*** for the next bounding box
		if(!current->isLeaf()){
			BinTreeNode* leftChild = current->getLeftChild();
			BinTreeNode* rightChild = current->getRightChild();
			
			bool leftContainsRay = leftChild->containsRay(ray);
			bool rightContainsRay = rightChild->containsRay(ray);

			if(leftContainsRay && rightContainsRay){
				current = leftChild;
				collisionPotentials.add(rightChild);
			}else if(leftContainsRay){
				current = leftChild;
			}else if(rightContainsRay){
				current = rightChild;
			}else{
				//should never technically occur
				if(collisionPotentials.isEmpty()){
					break;
				}else{
					current = collisionPotentials.pop();
				}
			}
		}else{
			if(collisionPotentials.isEmpty()){
				break;
			}else{
				current = collisionPotentials.pop();
			}
		}
	}

	if(abs(tMin-CLIPPING_DISTANCE)>EPSILON){
		*mesh = currentBestMesh;
	}

	return tMin; 
}

__host__ __device__ BinTree::~BinTree(void){
	delete(root);
}
