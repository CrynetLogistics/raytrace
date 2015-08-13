#include "Ray.h"

#define BRIGHTNESS 8
#define SHADOW_DIM_FACTOR 0.5
#define CLIPPING_DISTANCE 999
#define MAX_BOUNCES 1
#define EPSILON 0.001

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
	while(rayType!=CLIPPING && rayNumber-1<MAX_BOUNCES){
		nextRayBounce();
	}

	//TODO: reduces intensity according to the number of bounces
	//pathColour.r = pathColour.r/rayNumber;
	//pathColour.g = pathColour.g/rayNumber;
	//pathColour.b = pathColour.b/rayNumber;

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

	colour_t intersectedMeshColour;
	if(tMin!=CLIPPING_DISTANCE){ //NOT a clipping ray
		intersectedMeshColour.r = scene->getMesh(iMin)->getColour().r;
		intersectedMeshColour.g = scene->getMesh(iMin)->getColour().g;
		intersectedMeshColour.b = scene->getMesh(iMin)->getColour().b;
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

	

	if(isShadowed){//backscatter by itself
		rayType = BACKSCATTER;
		pathColour.r /= 5;
		pathColour.g /= 5;
		pathColour.b /= 5;
		pathColour.r += (intersectedMeshColour.r*BRIGHTNESS*SHADOW_DIM_FACTOR/(rayNumber*totalDistance));
		pathColour.g += (intersectedMeshColour.g*BRIGHTNESS*SHADOW_DIM_FACTOR/(rayNumber*totalDistance));
		pathColour.b += (intersectedMeshColour.b*BRIGHTNESS*SHADOW_DIM_FACTOR/(rayNumber*totalDistance));
	}else{//continue to light - but path can be blocked by another object
		rayType = DIRECT;
		pathColour.r += (intersectedMeshColour.r*BRIGHTNESS/(rayNumber*totalDistance));
		pathColour.g += (intersectedMeshColour.g*BRIGHTNESS/(rayNumber*totalDistance));
		pathColour.b += (intersectedMeshColour.b*BRIGHTNESS/(rayNumber*totalDistance));
	}

	switch(rayType){
	case BACKSCATTER:
	{
		//reestablishes the normal the the point of intersection for next ray bounce
		//TODO: WHICH DIRECTION ARE THESE NORMALS POINTING??? - is this a source of error?
		ray = scene->getMesh(iMin)->getNormal(ray.getPosAtParameter(tMin), ray);
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
