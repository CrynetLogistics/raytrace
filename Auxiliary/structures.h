#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdint.h"

typedef struct vertex{
	float x;
	float y;
	float z;
} vertex_t;

//obsolete - use vector_t class instead
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

	//supersampling will be performed with a n by n grid
	//where n = MSAA_SAMPLES
	//MSAA_INDEX is 0 indexed
	int MSAA_INDEX;
	int MSAA_SAMPLES;
	int BSPBVH_DEPTH;
} launchParams_t;
