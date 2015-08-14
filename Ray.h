#pragma once
#include "vector_t.h"
#include "Scene.h"

//INITAL RAY FROM CAMERA
//DIRECT RAY IS UNOBSTRUCTED WITH THE OBJECT IT HAS INTERSECTED WITH
//BACKSCATTER RAY IS OBSTRUCTED WITH THE OBJECT IT HAS INTERSECTED WITH
//CLIPPING RAY DOES NOT UNDERGO FURTHER INTERSECTIONS
enum rayType_t {INITIAL, DIRECT, BACKSCATTER, CLIPPING};

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

	void nextRayBounce(void);
public:
	Ray(vector_t initial, Scene *scene, int MAX_BOUNCES);
	colour_t raytrace(void);
	~Ray(void);
};

