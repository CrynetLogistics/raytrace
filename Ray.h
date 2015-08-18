#pragma once
#include "Auxiliary/vector_t.h"
#include "Scene.h"

//INITAL RAY FROM CAMERA
//DIRECT RAY IS UNOBSTRUCTED WITH THE OBJECT IT HAS INTERSECTED WITH
//BACKSCATTER RAY IS OBSTRUCTED WITH THE OBJECT IT HAS INTERSECTED WITH
//CLIPPING RAY DOES NOT UNDERGO FURTHER INTERSECTIONS
//A RAY IS SET TO INITIAL AFTER BEING TRANSMISSION IN GLASS
enum rayType_t {INITIAL, DIRECT, BACKSCATTER, CLIPPING, TRANSMISSION};

class Ray
{
private:
	vector_t ray;
	colour_t pathColour;
	Scene *scene;
	//rayNumber = number of ray segments in scene from this ray
	int rayNumber;
	rayType_t rayType;
	float totalDistance;
	int MAX_BOUNCES;
	float currentMeshReflectivity;
	float specularityHighlight;
	int currentMeshIndex;

	__host__ __device__ void nextRayBounce(void);
public:
	__host__ __device__ Ray(vector_t initial, Scene *scene, int MAX_BOUNCES);
	__host__ __device__ colour_t raytrace(void);
	__host__ __device__ ~Ray(void);
};

