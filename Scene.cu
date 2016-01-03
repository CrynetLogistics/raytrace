#include "Scene.h"

//default lamp
__host__ __device__ Scene::Scene(int totalMeshes, uint32_t* textureData)
	:light(0, 0, 10, 10){
	this->totalMeshes = totalMeshes;
	this->textureData = textureData;
	
	meshes = (Mesh**)malloc(totalMeshes*sizeof(Mesh*));
	numOfMeshes = 0;
}

__host__ __device__ Mesh* Scene::getMesh(int number){
	if(number<=numOfMeshes){
		return meshes[number];
	}else{
		printf("ERROR: THAT MESH DOES NOT EXSIST, DEFAULT MESH RETURNED");
		return *(meshes+0);
	}
}

__host__ __device__ Camera Scene::getCamera(void){
	return camera;
}

__host__ __device__ void Scene::addSphere(float centreX, float centreY, float centreZ, float radius, colour_t col, materialType_t material){
	numOfMeshes++;

	Sphere *s;

	if(material!=TEXTURE){
		s = new Sphere(centreX, centreY, centreZ, radius, col, material);
	}else{
		s = new Sphere(centreX, centreY, centreZ, radius, col, textureData);
	}

	Mesh *m = s;
	meshes[numOfMeshes-1] = m;
}

__host__ __device__ void Scene::addPlane(vertex_t v1, vertex_t v2, vertex_t v3, vertex_t v4, colour_t colour, materialType_t material){
	numOfMeshes++;

	Plane *p;

	if(material!=TEXTURE){
		p = new Plane(v1, v2, v3, v4, colour, material);
	}else{
		p = new Plane(v1, v2, v3, v4, colour, textureData);
	}

	Mesh *m = p;
	meshes[numOfMeshes-1] = m;
}

__host__ __device__ void Scene::addTri(vertex_t v1, vertex_t v2, vertex_t v3, colour_t colour, materialType_t material){
	numOfMeshes++;

	Tri *t = new Tri(v1, v2, v3, colour, material);

	Mesh *m = t;
	meshes[numOfMeshes-1] = m;
}

__host__ __device__ void Scene::addLight(float posX, float posY, float posZ, float intensity){
	light = Light(posX, posY, posZ, intensity);
}

__host__ __device__ int Scene::getNumOfMeshes(void){
	return numOfMeshes;
}

__host__ __device__ Light Scene::getLight(void){
	return light;
}

__host__ __device__ uint32_t* Scene::getTexture(void){
	return textureData;
}

__host__ __device__ void Scene::setHorizonColour(colour_t horizonColour){
	this->horizonColour = horizonColour;
}

__host__ __device__ colour_t Scene::getHorizonColour(void){
	return horizonColour;
}

__host__ __device__ Mesh** Scene::getMeshes(void){
	return meshes;
}

//builds the binary space partioning bounding volume hierachy
__host__ void Scene::buildBSPBVH(int BSPBVH_DEPTH){
	//DEFINE MAX TREE HEIGHT AS 4
	partitioningHierachy = new BinTree(meshes, numOfMeshes);
	partitioningHierachy->buildTree(BSPBVH_DEPTH);
}

__device__ void Scene::buildBSPBVH(int BSPBVH_DEPTH, Stack<BinTreeNode*> *d_unPropagatedNodes){
	//DEFINE MAX TREE HEIGHT AS 4
	partitioningHierachy = new BinTree(meshes, numOfMeshes);
	partitioningHierachy->buildTree(BSPBVH_DEPTH, d_unPropagatedNodes);
}

__host__ __device__ float Scene::collisionDetect(vector_t ray, Mesh** mesh){
	return partitioningHierachy->findCollisionMesh(ray, mesh);
}

__host__ __device__ Scene::~Scene(void){
	for(int i=0;i<numOfMeshes;i++){
		free(meshes[i]);
	}
	free(meshes);
	delete(partitioningHierachy);
}
