#include "Auxiliary/structures.h"

typedef struct triPrototype{
	int v1, v2, v3;
} triPrototype_t;

typedef struct scenePrototype{
	vertex_t* verts;
	triPrototype_t* tris;
	int numOfTris;
	int numOfVerts;
} scenePrototype_t;

scenePrototype_t parseFile(std::string filename, int DEBUG_LEVEL);