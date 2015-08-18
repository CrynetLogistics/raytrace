#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum materialType_t{GLASS, WATER, DIFFUSE, SHINY};

class Material
{
private:
	float reflectivity;
	bool transmission;
public:
	__host__ __device__ float getReflectivity(void);
	__host__ __device__ bool getTransmission(void);
	__host__ __device__ Material(materialType_t material);
	__host__ __device__ ~Material(void);
};

