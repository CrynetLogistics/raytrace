#pragma once

enum materialType_t{GLASS, WATER, DIFFUSE, SHINY};

class Material
{
private:
	float reflectivity;
	bool transmission;
public:
	float getReflectivity(void);
	bool getTransmission(void);
	Material(materialType_t material);
	~Material(void);
};

