#include "Material.h"

__host__ __device__ Material::Material(materialType_t material)
{
	switch(material){
	case GLASS:
		reflectivity = 1;
		transmission = true;
		break;
	case WATER:
		reflectivity = 1;
		transmission = true;
		break;
	case DIFFUSE:
		reflectivity = 0;
		transmission = false;
		break;
	case SHINY:
		reflectivity = 1;
		transmission = false;
		break;
	}
}

__host__ __device__ float Material::getReflectivity(void){
	return reflectivity;
}

__host__ __device__ bool Material::getTransmission(void){
	return transmission;
}

__host__ __device__ Material::~Material(void)
{
}
