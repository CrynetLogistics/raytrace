#include "Material.h"

Material::Material(materialType_t material)
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

float Material::getReflectivity(void){
	return reflectivity;
}

bool Material::getTransmission(void){
	return transmission;
}

Material::~Material(void)
{
}
