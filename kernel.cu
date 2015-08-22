#include <iostream>
#include "SDL.h"
#include "stdio.h"
#include "math.h"
#include "Scene.h"
#include "Auxiliary/structures.h"
#include "Auxiliary/vector_t.h"
#include "Objects/Plane.h"
#include "Ray.h"
#include "Auxiliary/Material.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#undef main
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
#define RENDER_SQUARE_SIZE 40
#define DISPLAY_TIME 30000
#define MAX_ITERATIONS 4
#define THREADS_PER_BLOCK 64
#define NUM_OF_BLOCKS 25

void drawPixelRaytracer(SDL_Renderer *renderer, int x, int y, int squareSize);

int main()
{
    SDL_Window* window = NULL;
	SDL_Init(SDL_INIT_EVERYTHING);

	//create window
	window = SDL_CreateWindow("Raytracer", 
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    
	SDL_Renderer *renderer = NULL;
	renderer = SDL_CreateRenderer(window, 0, SDL_RENDERER_ACCELERATED);
	//BACKGROUND COLOUR SET
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderClear(renderer);


	for(int j=0; j<SCREEN_HEIGHT/RENDER_SQUARE_SIZE; j++){
		for(int i=0; i<SCREEN_WIDTH/RENDER_SQUARE_SIZE; i++){
			//CALL OUR DRAW LOOP FUNCTION
			drawPixelRaytracer(renderer, i, j, RENDER_SQUARE_SIZE);
			SDL_RenderPresent(renderer);
		}
	}



	printf("done");

	SDL_Delay(DISPLAY_TIME);
	//Destroy window
    SDL_DestroyWindow(window);
    //Quit SDL subsystems
    SDL_Quit();
    return 0;
}

__global__ void cudaShootRays(vector_t* lightRay, colour_t* colGrid){

	colour_t dark_red;
	dark_red.r = 150;
	dark_red.g = 0;
	dark_red.b = 0;
	colour_t soft_red;
	soft_red.r = 255;
	soft_red.g = 77;
	soft_red.b = 99;
	colour_t bright_green;
	bright_green.r = 33;
	bright_green.g = 255;
	bright_green.b = 108;
	colour_t cold_blue;
	cold_blue.r = 12;
	cold_blue.g = 37;
	cold_blue.b = 255;

	vertex_t v1;
	vertex_t v2;
	vertex_t v3;
	vertex_t v4;
	vertex_t v5;
	vertex_t v6;
	vertex_t v7;
	vertex_t v8;

	v1.x = -30;v1.y = 0;v1.z = -3;
	v2.x = 30;v2.y = 0;v2.z = -3;
	v3.x = -30;v3.y = 50;v3.z = -3;
	v4.x = 30;v4.y = 50;v4.z = -3;
	v5.x = -30;v5.y = 50;v5.z = 30;
	v6.x = 30;v6.y = 50;v6.z = 30;
	v7.x = -30;v7.y = 0;v7.z = 30;
	v8.x = 30;v8.y = 0;v8.z = 30;

	//6 Meshes; Meshes = {Spheres, Planes}
	Scene scene(9);
	scene.addLight(-1,8,6,10);
	scene.addPlane(v1,v2,v3,v4,bright_green,SHINY);
	//scene.addPlane(v3,v4,v5,v6,bright_green,SHINY);
	scene.addTri(v3,v4,v5,bright_green,SHINY);
	//scene.addPlane(v7,v8,v5,v6,bright_green,DIFFUSE);
	//scene.addPlane(v1,v3,v5,v7,bright_green,DIFFUSE);
	//scene.addPlane(v2,v4,v6,v8,bright_green,DIFFUSE);
	scene.addSphere(2,10,5,2.5,dark_red,SHINY);
	scene.addSphere(6,9,3,3,cold_blue,DIFFUSE);
	scene.addSphere(6,7,-1,2,cold_blue,SHINY);
	scene.addSphere(-2,6,0,2,soft_red,WATER);
	scene.addSphere(-6,8,-2,2,soft_red,GLASS);
	scene.addSphere(-9,8,3,3,bright_green,DIFFUSE);



	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Ray ray(lightRay[index], &scene, MAX_ITERATIONS);
	colGrid[index] = ray.raytrace();
}

//where x and y are the top left most coordinates and squareSize is one block being rendered
void drawPixelRaytracer(SDL_Renderer *renderer, int x, int y, int squareSize){
	SDL_Rect r;
	r.h = 1;
	r.w = 1;

	//vector_t locDir = scene->getCamera().getLocDir();
	vector_t locDir(0,0,0,0,3,0);
	//float ZOOM_FACTOR = scene->getCamera().getGridSize();
	float ZOOM_FACTOR = 0.01f;

	vector_t* thisLocDir = (vector_t*)malloc(squareSize*squareSize*sizeof(vector_t));
	colour_t* col = (colour_t*)malloc(squareSize*squareSize*sizeof(colour_t));

	for(int i=x*squareSize;i<(x+1)*squareSize;i++){
		for(int j=y*squareSize;j<(y+1)*squareSize;j++){
			int index = (j-y*squareSize)*squareSize+(i-x*squareSize);
			//TODO: GENERALISE THIS FORMULA
			//CURRENTLY: ONLY APPLIES TO CAMERA POINTING ALONG Y DIRECTION
			thisLocDir[index] = vector_t();
			thisLocDir[index].x0 = locDir.x0;
			thisLocDir[index].y0 = locDir.y0;
			thisLocDir[index].z0 = locDir.z0;

			thisLocDir[index].xt = locDir.xt + (float)(i-SCREEN_WIDTH/2)*ZOOM_FACTOR;
			thisLocDir[index].yt = locDir.yt;
			thisLocDir[index].zt = locDir.zt + (float)(SCREEN_HEIGHT/2-j)*ZOOM_FACTOR;
		}
	}



	//TODO:modify code to exploit dim3 struct of kernel call instead of using a linear array
	//CALL PARALLEL CUDA ALGORITHM HERE
	vector_t* d_lightRay;
	colour_t* d_colourGrid;

	cudaMalloc((void**) &d_lightRay, sizeof(vector_t)*squareSize*squareSize);
	cudaMalloc((void**) &d_colourGrid, sizeof(colour_t)*squareSize*squareSize);

	cudaMemcpy(d_lightRay, thisLocDir, sizeof(vector_t)*squareSize*squareSize, cudaMemcpyHostToDevice);

	//calculateIntensityFromIntersections(thisLocDir, scene, col, squareSize*squareSize);
	//CURRENTLY SPECIFIC TO 80 BLOCKS CHANGE THIS LOL
	cudaShootRays<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(d_lightRay, d_colourGrid);

	cudaMemcpy(col, d_colourGrid, sizeof(colour_t)*squareSize*squareSize, cudaMemcpyDeviceToHost);


	cudaFree(d_lightRay);
	cudaFree(d_colourGrid);
	//END OF GPU CALLING CUDA CODE


	for(int i=x*squareSize;i<(x+1)*squareSize;i++){
		for(int j=y*squareSize;j<(y+1)*squareSize;j++){
			int index = (j-y*squareSize)*squareSize+(i-x*squareSize);
			r.x = i;
			r.y = j;

			if(col[index].r<=255 && col[index].g<=255 && col[index].b<=255){
				SDL_SetRenderDrawColor(renderer, (int)col[index].r, (int)col[index].g, (int)col[index].b, 255);
			}else{
				//draw bright flourescent pink for regions out of colour range nice one zl
				SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
			}
			
			SDL_RenderFillRect(renderer, &r);
		}
	}

	free(thisLocDir);
	free(col);
}