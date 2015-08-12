#include "Ray.h"
#include "Mesh.h"
#include "stdio.h"

#define BRIGHTNESS 5
#define SHADOW_DIM_FACTOR 0.6
#define CLIPPING_DISTANCE 999
#define MAX_BOUNCES 20

//TODO: NEED TO CORRECT FLOATING POINT ERRORS IN CALCULATIONS
//THESE ERRORS ARE CAUSING INTERSECTION ERRORS ON EITHER SIDE OF A SURFACE
Ray::Ray(vector_t initial, Scene *scene)
{
	ray = initial;
	this->scene = scene;
	pathColour.r = 0;
	pathColour.g = 0;
	pathColour.b = 0;
	rayNumber = 0;
	rayType = INITIAL;
	totalDistance = 0;
}

colour_t Ray::raytrace(void){
	while(rayType!=CLIPPING && rayNumber<MAX_BOUNCES){
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
		//getSphere will become getGENERAL with GENERAL encompassing spheres, planes etc
		tCurrent = scene->getMesh(i)->getIntersectionParameter(ray, scene->getLight());
		if(tCurrent<tMin && tCurrent!=0){
			tMin = tCurrent;
			iMin = i;
		}
	}

	colour_t intersectedMeshColour;
	if(tMin!=CLIPPING_DISTANCE){ //NOT a clipping ray
		intersectedMeshColour.r = (float)scene->getMesh(iMin)->getColour().r;
		intersectedMeshColour.g = (float)scene->getMesh(iMin)->getColour().g;
		intersectedMeshColour.b = (float)scene->getMesh(iMin)->getColour().b;
	}else{
		pathColour.r /= 2;
		pathColour.g /= 2;
		pathColour.b /= 2;
		rayType = CLIPPING;
		return;
	}

	float distance = ray.calculateDistance(tMin);
	totalDistance += distance;
	bool isShadowed = scene->getMesh(iMin)->getShadowedStatus(ray, tMin, scene->getLight());

	

	if(isShadowed){
		rayType = BACKSCATTER;
		pathColour.r += (int)(intersectedMeshColour.r*BRIGHTNESS*SHADOW_DIM_FACTOR/(rayNumber*totalDistance));
		pathColour.g += (int)(intersectedMeshColour.g*BRIGHTNESS*SHADOW_DIM_FACTOR/(rayNumber*totalDistance));
		pathColour.b += (int)(intersectedMeshColour.b*BRIGHTNESS*SHADOW_DIM_FACTOR/(rayNumber*totalDistance));
	}else{
		rayType = DIRECT;
		pathColour.r += (int)(intersectedMeshColour.r*BRIGHTNESS/(rayNumber*totalDistance));
		pathColour.g += (int)(intersectedMeshColour.g*BRIGHTNESS/(rayNumber*totalDistance));
		pathColour.b += (int)(intersectedMeshColour.b*BRIGHTNESS/(rayNumber*totalDistance));
	}

	switch(rayType){
	case BACKSCATTER:
	{
		//reestablishes the normal the the point of intersection for next ray bounce
		//TODO: WHICH DIRECTION ARE THESE NORMALS POINTING???
		ray = scene->getMesh(iMin)->getNormal(ray.getPosAtParameter(tMin));
		break;
	}
	case DIRECT:
	{
		vertex_t pos = ray.getPosAtParameter(tMin);
		ray.x0 = pos.x;
		ray.y0 = pos.y;
		ray.z0 = pos.z;
		vertex_t lightPos = scene->getLight().getPos();
		ray.xt = lightPos.x - pos.x;
		ray.yt = lightPos.y - pos.y;
		ray.zt = lightPos.z - pos.z;
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
