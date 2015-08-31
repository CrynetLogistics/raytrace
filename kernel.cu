#include <iostream>
#include "SDL.h"
#include "SDL_image.h"
#include "stdio.h"
#include "stdint.h"
#include "math.h"
#include "Scene.h"
#include "Auxiliary/structures.h"
#include "Auxiliary/vector_t.h"
#include "Plane.h"
#include "Ray.h"
#include "Material.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#undef main
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
#define RENDER_SQUARE_SIZE 80
#define DISPLAY_TIME 30000
#define MAX_ITERATIONS 4
#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 25
#define TEXTURE_WIDTH 600
#define TEXTURE_HEIGHT 300

uint32_t getpixel(SDL_Surface *surface, int x, int y);

__global__ void d_initScene(Scene* d_scene, uint32_t* textureData){
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
	colour_t black;
	black.r = 0;
	black.g = 0;
	black.b = 0;

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
	//Scene scene(9, textureData);
	d_scene = new (d_scene) Scene(9, textureData);
	//d_scene = new Scene(9, textureData);
	d_scene->addLight(-1,8,6,10);
	d_scene->setHorizonColour(black);
	d_scene->addPlane(v1,v2,v3,v4,bright_green,SHINY);
	//scene->addPlane(v3,v4,v5,v6,bright_green,SHINY);
	d_scene->addTri(v3,v4,v5,bright_green,SHINY);
	//scene->addPlane(v7,v8,v5,v6,bright_green,DIFFUSE);
	//scene->addPlane(v1,v3,v5,v7,bright_green,DIFFUSE);
	//scene->addPlane(v2,v4,v6,v8,bright_green,DIFFUSE);
	d_scene->addSphere(2,10,5,2.5,dark_red,SHINY);
	d_scene->addSphere(6,9,3,3,cold_blue,DIFFUSE);
	d_scene->addSphere(6,7,-1,2,cold_blue,SHINY);
	d_scene->addSphere(-2,6,0,1.2,soft_red,WATER);
	d_scene->addSphere(-6,8,-2,2,soft_red,GLASS);
	d_scene->addSphere(-9,8,3,3,bright_green,SHINY);
}

__global__ void cudaShootRays(vector_t* lightRay, colour_t* colGrid, Scene* d_scene){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Ray ray(lightRay[index], d_scene, MAX_ITERATIONS);
	colGrid[index] = ray.raytrace();
}

Scene* initScene(uint32_t* h_texture){
	Scene* d_scene;
	uint32_t* d_textureData;
	
	cudaMalloc((void**) &d_textureData, sizeof(uint32_t)*TEXTURE_HEIGHT*TEXTURE_WIDTH);
	cudaMalloc((void**) &d_scene, sizeof(Scene));
	
	cudaMemcpy(d_textureData, h_texture, sizeof(uint32_t)*TEXTURE_HEIGHT*TEXTURE_WIDTH, cudaMemcpyHostToDevice);

	d_initScene<<<1,1>>>(d_scene, d_textureData);

	return d_scene;
}

void destroyScene(Scene* d_scene, uint32_t* d_textureData){
	cudaFree(d_textureData);
	cudaFree(d_scene);
}

//where x and y are the top left most coordinates and squareSize is one block being rendered
void drawPixelRaytracer(SDL_Renderer *renderer, int x, int y, int squareSize, Scene* d_scene){

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




	//CALL PARALLEL CUDA ALGORITHM HERE
	vector_t* d_lightRay;
	colour_t* d_colourGrid;

	cudaMalloc((void**) &d_lightRay, sizeof(vector_t)*squareSize*squareSize);
	cudaMalloc((void**) &d_colourGrid, sizeof(colour_t)*squareSize*squareSize);

	cudaMemcpy(d_lightRay, thisLocDir, sizeof(vector_t)*squareSize*squareSize, cudaMemcpyHostToDevice);

	//calculateIntensityFromIntersections(thisLocDir, scene, col, squareSize*squareSize);
	//CURRENTLY SPECIFIC TO 80 BLOCKS CHANGE THIS LOL
	cudaShootRays<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(d_lightRay, d_colourGrid, d_scene);

	cudaMemcpy(col, d_colourGrid, sizeof(colour_t)*squareSize*squareSize, cudaMemcpyDeviceToHost);


	cudaFree(d_lightRay);
	cudaFree(d_colourGrid);
	//END OF GPU CALLING CUDA CODE


	for(int i=x*squareSize;i<(x+1)*squareSize;i++){
		for(int j=y*squareSize;j<(y+1)*squareSize;j++){
			int index = (j-y*squareSize)*squareSize+(i-x*squareSize);

			if(col[index].r<=255 && col[index].g<=255 && col[index].b<=255){
				SDL_SetRenderDrawColor(renderer, (int)col[index].r, (int)col[index].g, (int)col[index].b, 255);
			}else{
				//draw bright flourescent pink for regions out of colour range nice one zl
				SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
			}
			SDL_RenderDrawPoint(renderer, i, j);
		}
	}

	free(thisLocDir);
	free(col);
}

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


	//////////////START OF TEXTURE LOAD
	SDL_Surface* texture;
	texture = IMG_Load("texture.png");
	if(!texture){
		printf("ERROR:%s", IMG_GetError());
	}


	uint32_t* tData = (uint32_t*)malloc(TEXTURE_WIDTH*TEXTURE_HEIGHT*sizeof(uint32_t));
	for(int i=0;i<TEXTURE_WIDTH*TEXTURE_HEIGHT;i++){
		tData[i] = getpixel(texture, i%TEXTURE_WIDTH, i/TEXTURE_WIDTH);
	}
	//END OF TEXTURE LOAD


	Scene* d_scene = initScene(tData);


	for(int j=0; j<SCREEN_HEIGHT/RENDER_SQUARE_SIZE; j++){
		for(int i=0; i<SCREEN_WIDTH/RENDER_SQUARE_SIZE; i++){
			//CALL OUR DRAW LOOP FUNCTION
			drawPixelRaytracer(renderer, i, j, RENDER_SQUARE_SIZE, d_scene);
			SDL_RenderPresent(renderer);
		}
	}
	


	printf("done");

	destroyScene(d_scene, tData);
	IMG_Quit();
	free(tData);
	SDL_Delay(DISPLAY_TIME);
	//Destroy window
    SDL_DestroyWindow(window);
    //Quit SDL subsystems
    SDL_Quit();
    return 0;
}

//UTILITY FUNCTION COURTESY OF sdl.beuc.net/sdl.wiki/Pixel_Access

uint32_t getpixel(SDL_Surface *surface, int x, int y)
{
    int bpp = surface->format->BytesPerPixel;
    /* Here p is the address to the pixel we want to retrieve */
    uint8_t *p = (uint8_t *)surface->pixels + y * surface->pitch + x * bpp;

    switch(bpp) {
    case 1:
        return *p;
        break;

    case 2:
        return *(uint16_t *)p;
        break;

    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN)
            return p[0] << 16 | p[1] << 8 | p[2];
        else
            return p[0] | p[1] << 8 | p[2] << 16;
        break;

    case 4:
        return *(uint32_t *)p;
        break;

    default:
        return 0;       /* shouldn't happen, but avoids warnings */
    }
}