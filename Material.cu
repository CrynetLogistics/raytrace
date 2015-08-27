#include "Material.h"
#include "stdio.h"

__host__ __device__ Material::Material(){

}

__host__ __device__ void Material::initMaterial(materialType_t material){
	switch(material){
	case GLASS:
		reflectivity = 1;
		transmission = true;
		hasTextureMapping = 0;
		break;
	case WATER:
		reflectivity = 1;
		transmission = true;
		hasTextureMapping = 0;
		break;
	case DIFFUSE:
		reflectivity = 0;
		transmission = false;
		hasTextureMapping = 0;
		break;
	case SHINY:
		reflectivity = 1;
		transmission = false;
		hasTextureMapping = 0;
		break;
	}
}

__host__ __device__ void Material::initMaterial(uint32_t* textureData){
	this->textureData = textureData;
	reflectivity = 0;
	transmission = false;
	hasTextureMapping = 2;
}

__host__ __device__ float Material::getReflectivity(void){
	return reflectivity;
}

__host__ __device__ bool Material::getTransmission(void){
	return transmission;
}

__host__ __device__ int Material::isTextured(void){
	return hasTextureMapping;
}

__host__ __device__ uint32_t* Material::getTexture(void){
	return textureData;
}

__host__ __device__ Material::~Material(void){
}
