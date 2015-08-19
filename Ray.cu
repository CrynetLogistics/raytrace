#include "Ray.h"

#define BRIGHTNESS 4
#define CLIPPING_DISTANCE 999
#define EPSILON 0.001f
//REFLECTIVITY, SHADOW_DIM_FACTOR GOES FROM 0 TO 1
#define REFLECTIVITY 0.8f
#define SHADOW_DIM_FACTOR 0.9f
#define REFRACTION_BRIGHTNESS 0.5f //[0,1] higher = more prominence of reflections and refractions
#define GLASS_CLARITY 0.6f //[0,1] higher = less of original colour
#define IOR 1.5f
#define REFLECTIONAL_LIGHTING 1
#define MAXIMUM_DEPTH 8

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
__device__ Ray::Ray(vector_t initial, Scene *scene, int MAX_BOUNCES)
{
	ray = initial;
	this->scene = scene;
	this->MAX_BOUNCES = MAX_BOUNCES;
	pathColour.r = 0;
	pathColour.g = 0;
	pathColour.b = 0;
	totalColour.r = 0;
	totalColour.g = 0;
	totalColour.b = 0;
	rayNumber = 0;
	rayType = INITIAL;
	totalDistance = 0;
	specularityHighlight = 0;
	secondary = (secondaryRay_t*)malloc(MAXIMUM_DEPTH*sizeof(secondaryRay_t));
	secondaryDepth = 0;
}

__device__ colour_t Ray::raytrace(void){
	int rayLevel = 0;
	while(rayLevel <= secondaryDepth){
		if(rayLevel != 0){
			ray = secondary[rayLevel-1].normalRaySecondary;
			pathColour.r = 0;
			pathColour.g = 0;
			pathColour.b = 0;
			rayNumber = 0;
			rayType = INITIAL;
			MAX_BOUNCES = secondary[rayLevel-1].MAX_BOUNCESSecondary;
		}

		while(rayType!=CLIPPING && rayNumber-1<MAX_BOUNCES){
			nextRayBounce();
		}

		//add specularity highlights if appropiate
		if(specularityHighlight>0.99){
			pathColour.r *= 2-(1-specularityHighlight)*100;
			pathColour.g *= 2-(1-specularityHighlight)*100;
			pathColour.b *= 2-(1-specularityHighlight)*100;
			specularityHighlight = 0;
		}

		if(rayLevel == 0){
			totalColour.r += pathColour.r;
			totalColour.g += pathColour.g;
			totalColour.b += pathColour.b;
		}else{
			totalColour.r += pathColour.r*REFLECTIVITY*secondary[rayLevel-1].currentMeshReflectivitySecondary;
			totalColour.g += pathColour.g*REFLECTIVITY*secondary[rayLevel-1].currentMeshReflectivitySecondary;
			totalColour.b += pathColour.b*REFLECTIVITY*secondary[rayLevel-1].currentMeshReflectivitySecondary;
		}
		rayLevel++;
	}
	return totalColour;
}

__device__ void Ray::nextRayBounce(void){
	rayNumber++;
	float tMin = CLIPPING_DISTANCE;
	int iMin = 0;

	for(int i=0;i<scene->getNumOfMeshes();i++){

		float tCurrent = scene->getMesh(i)->getIntersectionParameter(ray);

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
		rayType = CLIPPING;
		return;
	}

	float distance = ray.calculateDistance(tMin);
	totalDistance += distance;
	
	float currentMeshReflectivity = scene->getMesh(iMin)->getMaterial().getReflectivity();
	bool isTransmission = scene->getMesh(iMin)->getMaterial().getTransmission();
	float isShadowed = scene->getMesh(iMin)->getShadowedStatus(ray, tMin, scene->getLight());

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
	}else if(rayType==TRANSMISSION){
		//add nothing for now

		//also needs to force TRANSMISSION off when light is exiting
		isTransmission = false;
		rayType = BACKSCATTER;
		pathColour.r *= 1.5+2*REFRACTION_BRIGHTNESS;
		pathColour.g *= 1.5+2*REFRACTION_BRIGHTNESS;
		pathColour.b *= 1.5+2*REFRACTION_BRIGHTNESS;
		//////////////////////////////FORCE LEAVE GLASS MATERIAL REVERSE SNELLS LAW
		vertex_t pos = ray.getPosAtParameter(tMin);
		vector_t transmissionRay;
		vector_t normalRay = scene->getMesh(iMin)->getNormal(ray.getPosAtParameter(tMin), ray);
		vector_t reverseIncidence(0,0,0,-ray.xt,-ray.yt,-ray.zt);
		float cosINCIDENCE = normalRay.directionDotProduct(reverseIncidence);
		float normalMagnitude = normalRay.directionMagnitude();
		float rayMagnitude = ray.directionMagnitude();
		float hypFactor = (cosINCIDENCE + sqrt(cosINCIDENCE*cosINCIDENCE+(1/IOR)*(1/IOR)-1))/((1/IOR)*(1/IOR)-1);
		transmissionRay.x0 = pos.x;
		transmissionRay.y0 = pos.y;
		transmissionRay.z0 = pos.z;
		transmissionRay.xt = normalRay.xt/normalMagnitude + ray.xt*hypFactor/rayMagnitude;
		transmissionRay.yt = normalRay.yt/normalMagnitude + ray.yt*hypFactor/rayMagnitude;
		transmissionRay.zt = normalRay.zt/normalMagnitude + ray.zt*hypFactor/rayMagnitude;

		ray = transmissionRay;
		return;
	}
	
	
	//UPDATE PRIMARY LIGHTING
	if(isTransmission){
		rayType = TRANSMISSION;
		pathColour.r -= (0.6+0.4*GLASS_CLARITY)*intersectedMeshColour.r*BRIGHTNESS/(rayNumber*totalDistance);
		pathColour.g -= (0.6+0.4*GLASS_CLARITY)*intersectedMeshColour.g*BRIGHTNESS/(rayNumber*totalDistance);
		pathColour.b -= (0.6+0.4*GLASS_CLARITY)*intersectedMeshColour.b*BRIGHTNESS/(rayNumber*totalDistance);
	}else if(isShadowed > 0){//backscatter by itself
		rayType = BACKSCATTER;//THIS CORRESPONDS TO THE REFLECTED RAY TYPE
		pathColour.r *= (1-currentMeshReflectivity/2)*(1-abs(isShadowed));
		pathColour.g *= (1-currentMeshReflectivity/2)*(1-abs(isShadowed));
		pathColour.b *= (1-currentMeshReflectivity/2)*(1-abs(isShadowed));
	}else{//continue to light - but path can be blocked by another object
		rayType = DIRECT;//THIS CORRESPONDS TO THE REFLECTED RAY TYPE
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

	//////////////////////FORCE ENTER GLASS MATERIAL SNELLS LAW
	vector_t transmissionRay;
	vector_t reverseIncidence(0,0,0,-ray.xt,-ray.yt,-ray.zt);
	float cosINCIDENCE = normalRay.directionDotProduct(reverseIncidence);
	float normalMagnitude = normalRay.directionMagnitude();
	float rayMagnitude = ray.directionMagnitude();
	float hypFactor = (cosINCIDENCE + sqrt(cosINCIDENCE*cosINCIDENCE+IOR*IOR-1))/(IOR*IOR-1);
	transmissionRay.x0 = pos.x;
	transmissionRay.y0 = pos.y;
	transmissionRay.z0 = pos.z;
	transmissionRay.xt = normalRay.xt/normalMagnitude + ray.xt*hypFactor/rayMagnitude;
	transmissionRay.yt = normalRay.yt/normalMagnitude + ray.yt*hypFactor/rayMagnitude;
	transmissionRay.zt = normalRay.zt/normalMagnitude + ray.zt*hypFactor/rayMagnitude;



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
		
		if(REFLECTIONAL_LIGHTING && secondaryDepth < MAXIMUM_DEPTH){
			secondary[secondaryDepth].currentMeshReflectivitySecondary = currentMeshReflectivity;
			secondary[secondaryDepth].MAX_BOUNCESSecondary = MAX_BOUNCES-1;
			secondary[secondaryDepth].normalRaySecondary = normalRay;
			secondaryDepth++;
		}

		//SPECULARITY
		specularityHighlight = normalRay.directionDotProduct(directRay)/(normalRay.directionMagnitude()*directRay.directionMagnitude());
		break;
	}
	case TRANSMISSION:
	{
		ray = transmissionRay;

		if(REFLECTIONAL_LIGHTING && secondaryDepth < MAXIMUM_DEPTH){
			secondary[secondaryDepth].currentMeshReflectivitySecondary = currentMeshReflectivity;
			secondary[secondaryDepth].MAX_BOUNCESSecondary = MAX_BOUNCES-1;
			secondary[secondaryDepth].normalRaySecondary = normalRay;
			secondaryDepth++;
		}

		//SPECULARITY
		float specularityHighlight = normalRay.directionDotProduct(directRay)/(normalRay.directionMagnitude()*directRay.directionMagnitude());
		break;
	}
	default:
		printf("ERROR: RAYS HAVE GONE HAYWIRE");
		break;
	}
}

__device__ Ray::~Ray(void)
{
	free(secondary);
}
