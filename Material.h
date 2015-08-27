#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdint.h"

enum materialType_t{GLASS, WATER, DIFFUSE, SHINY, TEXTURE};

class Material
{
private:
	float reflectivity;
	bool transmission;
	
	
public:
	uint32_t* textureData;
	int hasTextureMapping;
	__host__ __device__ float getReflectivity(void);
	__host__ __device__ bool getTransmission(void);
	__host__ __device__ int isTextured(void);
	__host__ __device__ uint32_t* getTexture(void);
	__host__ __device__ Material();
	__host__ __device__ void initMaterial(materialType_t material);
	__host__ __device__ void initMaterial(uint32_t* textureData);
	__host__ __device__ ~Material(void);
};

