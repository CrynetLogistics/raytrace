#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdint.h"

typedef struct vertex{
	float x;
	float y;
	float z;
} vertex_t;

//typedef struct vector{
//	float x0;
//	float y0;
//	float z0;
//	float xt;
//	float yt;
//	float zt;
//} vector_t;

typedef struct colour{
	float r;
	float g;
	float b;
} colour_t;

typedef struct launchParams{
	int x;
	int y;
	int squareSizeX;
	int squareSizeY;
	int SCREEN_X;
	int SCREEN_Y;
} launchParams_t;