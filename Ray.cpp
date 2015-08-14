#include "Ray.h"

#define BRIGHTNESS 4
#define CLIPPING_DISTANCE 999
#define EPSILON 0.001
//REFLECTIVITY, SHADOW_DIM_FACTOR GOES FROM 0 TO 1
#define REFLECTIVITY 0.5
#define SHADOW_DIM_FACTOR 0.5

/*
LIFE CYCLE OF A RAY:
starts off as an INITIAL ray and is predefined a start position and direction
the ray is then shot out, the intersection properties worked out and then,
given CLIPPING iff no intersections occur, else the vector from the point of
intersection to the lamp is calculated and if the normal of the surface blocks
the light from going to the camera then it is given a BACKSCATTER, else DIRECT
NOTE: Even if DIRECT is given to a ray, it can still encounter other objects
that might obstruct its path in future - in this case, it will be given a 
BACKSCATTER when that occurs
*/
Ray::Ray(vector_t initial, Scene *scene, int MAX_BOUNCES)
{
	ray = initial;
	this->scene = scene;
	this->MAX_BOUNCES = MAX_BOUNCES;
	pathColour.r = 0;
	pathColour.g = 0;
	pathColour.b = 0;
	rayNumber = 0;
	rayType = INITIAL;
	totalDistance = 0;
	currentMeshReflectivity = 1;
}

colour_t Ray::raytrace(void){
	while(rayType!=CLIPPING && rayNumber-1<MAX_BOUNCES){
		nextRayBounce();
	}
	return pathColour;
}

void Ray::nextRayBounce(void){
	rayNumber++;
	float tMin = CLIPPING_DISTANCE;
	float tCurrent = 0;
	int iMin = 0;

	for(int i=0;i<scene->getNumOfMeshes();i++){

		tCurrent = scene->getMesh(i)->getIntersectionParameter(ray, scene->getLight());

		if(EPSILON<tCurrent && tCurrent<tMin && tCurrent!=0){
			tMin = tCurrent;
			iMin = i;
		}
	}

	float intersectedMeshReflectivity;
	colour_t intersectedMeshColour;
	if(tMin!=CLIPPING_DISTANCE){ //NOT a clipping ray
		intersectedMeshColour.r = scene->getMesh(iMin)->getColour().r;
		intersectedMeshColour.g = scene->getMesh(iMin)->getColour().g;
		intersectedMeshColour.b = scene->getMesh(iMin)->getColour().b;
	}else{
		rayType = CLIPPING;
		return;
	}

	float distance = ray.calculateDistance(tMin);
	totalDistance += distance;
	bool isShadowed = scene->getMesh(iMin)->getShadowedStatus(ray, tMin, scene->getLight());

	if(rayType==BACKSCATTER){
		pathColour.r += intersectedMeshColour.r*BRIGHTNESS*currentMeshReflectivity/(rayNumber*totalDistance);
		pathColour.g += intersectedMeshColour.g*BRIGHTNESS*currentMeshReflectivity/(rayNumber*totalDistance);
		pathColour.b += intersectedMeshColour.b*BRIGHTNESS*currentMeshReflectivity/(rayNumber*totalDistance);
	}else if(rayType==DIRECT){
		pathColour.r += intersectedMeshColour.r*BRIGHTNESS/(rayNumber*totalDistance);
		pathColour.g += intersectedMeshColour.g*BRIGHTNESS/(rayNumber*totalDistance);
		pathColour.b += intersectedMeshColour.b*BRIGHTNESS/(rayNumber*totalDistance);
	}else if(rayType==INITIAL){
		pathColour.r += intersectedMeshColour.r*BRIGHTNESS/(rayNumber*totalDistance);
		pathColour.g += intersectedMeshColour.g*BRIGHTNESS/(rayNumber*totalDistance);
		pathColour.b += intersectedMeshColour.b*BRIGHTNESS/(rayNumber*totalDistance);
	}
	
	currentMeshReflectivity = scene->getMesh(iMin)->getReflectivity();

	//UPDATE PRIMARY LIGHTING
	if(isShadowed){//backscatter by itself
		rayType = BACKSCATTER;//THIS CORRESPONDS TO THE REFLECTED RAY TYPE
		pathColour.r /= 10*(1-SHADOW_DIM_FACTOR);
		pathColour.g /= 10*(1-SHADOW_DIM_FACTOR);
		pathColour.b /= 10*(1-SHADOW_DIM_FACTOR);
		
		
	}else{//continue to light - but path can be blocked by another object
		rayType = DIRECT;//THIS CORRESPONDS TO THE REFLECTED RAY TYPE
		//pathColour.r += intersectedMeshColour.r*BRIGHTNESS/(rayNumber*totalDistance);
		//pathColour.g += intersectedMeshColour.g*BRIGHTNESS/(rayNumber*totalDistance);
		//pathColour.b += intersectedMeshColour.b*BRIGHTNESS/(rayNumber*totalDistance);
	}



	vector_t normalRay = scene->getMesh(iMin)->getNormal(ray.getPosAtParameter(tMin), ray);
	vector_t directRay;
	vertex_t pos = ray.getPosAtParameter(tMin);
	directRay.x0 = pos.x;
	directRay.y0 = pos.y;
	directRay.z0 = pos.z;
	vertex_t lightPos = scene->getLight().getPos();
	directRay.xt = lightPos.x - pos.x;
	directRay.yt = lightPos.y - pos.y;
	directRay.zt = lightPos.z - pos.z;



	//SET RAY FOR NEXT SCATTER AND PREPROCESS SECONDARY LIGHTING FOR NEXT RAY
	switch(rayType){
	case BACKSCATTER:
	{
		//reestablishes the normal the the point of intersection for next ray bounce
		//TODO: WHICH DIRECTION ARE THESE NORMALS POINTING??? - is this a source of error?
		ray = normalRay;

		break;
	}
	case DIRECT:
	{
		ray = directRay;

		Ray secondaryNormalRay(normalRay, scene, MAX_BOUNCES-1);
		colour_t secondaryColour = secondaryNormalRay.raytrace();
		pathColour.r += secondaryColour.r*REFLECTIVITY*currentMeshReflectivity;
		pathColour.g += secondaryColour.g*REFLECTIVITY*currentMeshReflectivity;
		pathColour.b += secondaryColour.b*REFLECTIVITY*currentMeshReflectivity;
		break;
	}
	default:
		printf("ERROR: RAYS HAVE GONE HAYWIRE");
		break;
	}
}

Ray::~Ray(void)
{
}
