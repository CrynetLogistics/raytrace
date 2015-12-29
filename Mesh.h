#pragma once
#include "Light.h"
#include "Auxiliary/vector_t.h"
#include "Material.h"
#include "Auxiliary/extremum_t.cuh"

class Mesh
{
public:
	__host__ __device__ Mesh(void);
	__host__ __device__ virtual float getIntersectionParameter(vector_t lightRay) = 0;
	__host__ __device__ virtual colour_t getColour(vertex_t position) = 0;
	__host__ __device__ virtual float getShadowedStatus(vector_t lightRay, float t, Light light) = 0;
	__host__ __device__ virtual vector_t getNormal(vertex_t pos, vector_t incoming) = 0;
	__host__ __device__ virtual Material getMaterial(void) = 0;
	
	//        {0}            {    not    }
	//returns {1} if Mesh is { partially } contained in volume
	//        {2}            {   fully   }
	//WARNING:This is designed for use with the BSP BVH structure implementation
	//        and is only correct if both nodes of the BTree are tested.
	//        If a Mesh is fully in one volume, then it is guarenteed not to be in the other
	//        If a Mesh is partially in one volume, then it could be in the other dispite the
	//        result of the other box.
	//        If a Mesh is partially in a parent and the nodes show that they have {partially, not}
	//        then the 'not' node must not be further subdivided.
	__host__ __device__ virtual int isContainedWithin(vertex_t extremum1, vertex_t extremum2) = 0;

	__host__ __device__ virtual extremum_t findExtremum(void) = 0;
	__host__ __device__ ~Mesh(void);
};
