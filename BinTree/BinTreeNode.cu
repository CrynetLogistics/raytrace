#include "BinTreeNode.cuh"

#define REPETITION_INDEX_SENSITIVITY 20
#define SMALLEST_MESH_NO_IN_BOUNDING_BOX 3

__host__ __device__ BinTreeNode::BinTreeNode(Mesh** meshes, int numOfMeshes){
	this->numOfMeshes = numOfMeshes;
	INITIAL_MESHES = numOfMeshes;
	this->meshes = (Mesh**)malloc(sizeof(Mesh*)*numOfMeshes);
	for(int i=0; i<numOfMeshes; i++){
		this->meshes[i] = meshes[i];
	}

	extremum_t init = meshes[0]->findExtremum();
	extremum = extremum_t(&init);
	for(int i=1; i<numOfMeshes; i++){
		extremum.mergeExtrema(meshes[i]->findExtremum());
	}

	vertex_t lowE = extremum.getLowExtremum();
	vertex_t highE = extremum.getHighExtremum();
	printf("Box, %i meshes, l[%.2f,%.2f,%.2f], h[%.2f,%.2f,%.2f]\n", numOfMeshes, lowE.x, lowE.y, lowE.z, highE.x, highE.y, highE.z);
	repetitionIndex = 0;
}

__host__ __device__ BinTreeNode::BinTreeNode(Mesh** meshes, int numOfMeshes, extremum_t extremum, int repetitionIndex){
	this->numOfMeshes = numOfMeshes;
	INITIAL_MESHES = numOfMeshes;
	this->meshes = (Mesh**)malloc(sizeof(Mesh*)*numOfMeshes);
	for(int i=0; i<numOfMeshes; i++){
		this->meshes[i] = meshes[i];
	}

	this->extremum = extremum;
	this->repetitionIndex = repetitionIndex;

	vertex_t lowE = extremum.getLowExtremum();
	vertex_t highE = extremum.getHighExtremum();
	printf("Box, %i meshes, l[%.2f,%.2f,%.2f], h[%.2f,%.2f,%.2f]\n", numOfMeshes, lowE.x, lowE.y, lowE.z, highE.x, highE.y, highE.z);
}

//The treeNode stops having leaves if it contains less than 3 meshes
//or if any single mesh partially intersects either or both of the
//meshes without having an anchoring vertex contained in at least
//one of the bounding boxes
//Termination of propagation also occurs if the tree height is greater
//than a specified length
__host__ __device__ void BinTreeNode::propagateTree(int maxTreeHeight){
	if(numOfMeshes < SMALLEST_MESH_NO_IN_BOUNDING_BOX || maxTreeHeight == 0){
		hasChildren = false;
		printf("STARTED WITH %i MESHES, FINISHED WITH %i MESHES\n", INITIAL_MESHES, numOfMeshes);
		return;
	}

	int currentHeadMesh = 0;
	int leftBoxMeshCount = 0;
	int rightBoxMeshCount = 0;
	Mesh** leftMeshes = (Mesh**)malloc(sizeof(Mesh*)*numOfMeshes);
	Mesh** rightMeshes = (Mesh**)malloc(sizeof(Mesh*)*numOfMeshes);
	extremum_t leftBox = extremum.getPartitionedLowExtremum();
	extremum_t rightBox = extremum.getPartitionedHighExtremum();

	for(int i=0; i<numOfMeshes; i++){
		int leftContainmentIndex = meshes[i]->isContainedWithin(leftBox);
		int rightContainmentIndex = meshes[i]->isContainedWithin(rightBox);
		if(leftContainmentIndex==2){
			leftMeshes[leftBoxMeshCount] = meshes[i];
			leftBoxMeshCount++;

			meshes[i] = meshes[numOfMeshes-1];
			numOfMeshes--;
			i--;
		}else if(rightContainmentIndex==2){
			rightMeshes[rightBoxMeshCount] = meshes[i];
			rightBoxMeshCount++;

			meshes[i] = meshes[numOfMeshes-1];
			numOfMeshes--;
			i--;
		}else{
			//leave the mesh in the parent and dont touch the children
		}
	}

	if(leftBoxMeshCount==INITIAL_MESHES || rightBoxMeshCount==INITIAL_MESHES){
		repetitionIndex++;
	}

	if(repetitionIndex>REPETITION_INDEX_SENSITIVITY){
		free(leftMeshes);
		free(rightMeshes);
		hasChildren = false;
		printf("STARTED WITH %i MESHES, FINISHED WITH %i MESHES\n", INITIAL_MESHES, numOfMeshes);
		return;
	}

	hasChildren = true;
	leftChild = new BinTreeNode(leftMeshes, leftBoxMeshCount, leftBox, repetitionIndex);
	rightChild = new BinTreeNode(rightMeshes, rightBoxMeshCount, rightBox, repetitionIndex);
	leftChild->propagateTree(maxTreeHeight-1);
	rightChild->propagateTree(maxTreeHeight-1);

	free(leftMeshes);
	free(rightMeshes);

	printf("STARTED WITH %i MESHES, FINISHED WITH %i MESHES\n", INITIAL_MESHES, numOfMeshes);
}

__host__ __device__ bool BinTreeNode::containsRay(vector_t ray){
	vertex_t minExtent = extremum.getLowExtremum();
	vertex_t maxExtent = extremum.getHighExtremum();

	float tNear, tFar;

	//X SLAB ANALYSIS
	//find perpendicular distances to two x slabs
	float xs1 = (minExtent.x - ray.x0)/ray.xt;
	float xs2 = (maxExtent.x - ray.x0)/ray.xt;
	if(xs1>xs2){
		tFar = xs1;
		tNear = xs2;
	}else{
		tNear = xs1;
		tFar = xs2;
	}

	//Y SLAB ANALYSIS
	//find perpendicular distances to two y slabs
	float ys1 = (minExtent.y - ray.y0)/ray.yt;
	float ys2 = (maxExtent.y - ray.y0)/ray.yt;
	if(ys1>ys2){
		if(ys1<tFar){
			tFar = ys1;
		}
		if(ys2>tNear){
			tNear = ys2;
		}
	}else{
		if(ys1>tNear){
			tNear = ys1;
		}
		if(ys2<tFar){
			tFar = ys2;
		}
	}

	//Test to see if a contradiction has occured yet
	if(tNear>tFar || tFar<0){
		return false;
	}

	//Z SLAB ANALYSIS
	//find perpendicular distances to two y slabs
	float zs1 = (minExtent.z - ray.z0)/ray.zt;
	float zs2 = (maxExtent.z - ray.z0)/ray.zt;
	if(zs1>zs2){
		if(zs1<tFar){
			tFar = zs1;
		}
		if(zs2>tNear){
			tNear = zs2;
		}
	}else{
		if(zs1>tNear){
			tNear = zs1;
		}
		if(zs2<tFar){
			tFar = zs2;
		}
	}

	if(tNear>tFar || tFar<0){
		return false;
	}else{
		return true;
	}
}

__host__ __device__ bool BinTreeNode::isLeaf(void){
	return !hasChildren;
}

__host__ __device__ BinTreeNode* BinTreeNode::getLeftChild(void){
	return leftChild;
}

__host__ __device__ BinTreeNode* BinTreeNode::getRightChild(void){
	return rightChild;
}

__host__ __device__ int BinTreeNode::getNumOfMeshes(void){
	return numOfMeshes;
}

__host__ __device__ Mesh* BinTreeNode::getMeshAt(int index){
	return meshes[index];
}

__host__ __device__ BinTreeNode::~BinTreeNode(void){
	if(hasChildren){
		delete(leftChild);
		delete(rightChild);
	}
	free(meshes);
}
