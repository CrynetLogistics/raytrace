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
#include "parser.h"

#undef main
//-----------------------------------------------------------------------------

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
//IF USE_BLOCK_BY_BLOCKING_RENDERING == 1
//	THEN RENDER_SQUARE_SIZE*RENDER_SQUARE_SIZE = THREADS_PER_BLOCK*NUM_OF_BLOCKS
//	ELSE SCREEN_WIDTH*SCREEN_HEIGHT = THREADS_PER_BLOCK*NUM_OF_BLOCKS
#define RENDER_SQUARE_SIZE 80
//
#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 25
//#define THREADS_PER_BLOCK 1024
//#define NUM_OF_BLOCKS 900
//
#define USE_CUDA 1
#define USE_BLOCK_BY_BLOCKING_RENDERING 1

//-----------------------------------------------------------------------------
#define MAX_ANIMATION_ITERATIONS 1 // 1 for just a still image
#define DISPLAY_TIME 30000
#define MAX_ITERATIONS 3
#define TEXTURE_WIDTH 600
#define TEXTURE_HEIGHT 300
//-----------------------------------------------------------------------------
uint32_t getpixel(SDL_Surface *surface, int x, int y);
void processColourOverspill(SDL_Renderer *renderer, colour_t col);
//-----------------------------------------------------------------------------

__global__ void d_initScene(Scene* d_scene, uint32_t* textureData, int* d_param,
							int* d_numOfTris, vertex_t* d_verts, triPrototype_t* d_tris){
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
	d_scene->addSphere(-2,6,0,1.2f,soft_red,WATER);
	d_scene->addSphere(*d_param-6,8,-2,2,soft_red,GLASS);
	d_scene->addSphere(-9,8,3,3,bright_green,SHINY);

	//auto parser

	for(int i=0; i<*d_numOfTris; i++){
		d_scene->addTri(d_verts[d_tris[i].v1-1], d_verts[d_tris[i].v2-1], d_verts[d_tris[i].v3-1], cold_blue, DIFFUSE);
	}

	//auto parser
}

Scene* d_initScene(uint32_t* h_texture, int t){
	Scene* d_scene;
	uint32_t* d_textureData;
	int* h_param = (int*)malloc(sizeof(int));
	*h_param = t;
	int* d_param;


	//auto parser

	scenePrototype_t exterior = parseFile();
	vertex_t* h_verts = exterior.verts;
	triPrototype_t* h_tris = exterior.tris;
	int h_numOfTris = exterior.numOfTris;
	int numOfVerts = exterior.numOfVerts;

	triPrototype_t* d_tris;
	vertex_t* d_verts;
	int* d_numOfTris;

	cudaMalloc((void**) &d_tris, h_numOfTris*sizeof(triPrototype_t));
	cudaMalloc((void**) &d_verts, numOfVerts*sizeof(vertex_t));
	cudaMalloc((void**) &d_numOfTris, sizeof(int));

	cudaMemcpy(d_tris, h_tris, h_numOfTris*sizeof(triPrototype_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_verts, h_verts, numOfVerts*sizeof(vertex_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_numOfTris, &h_numOfTris, sizeof(int), cudaMemcpyHostToDevice);

	//auto parser

	
	cudaMalloc((void**) &d_param, sizeof(int));
	cudaMalloc((void**) &d_textureData, sizeof(uint32_t)*TEXTURE_HEIGHT*TEXTURE_WIDTH);
	cudaMalloc((void**) &d_scene, sizeof(Scene));
	
	cudaMemcpy(d_param, h_param, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_textureData, h_texture, sizeof(uint32_t)*TEXTURE_HEIGHT*TEXTURE_WIDTH, cudaMemcpyHostToDevice);

	d_initScene<<<1,1>>>(d_scene, d_textureData, d_param, d_numOfTris, d_verts, d_tris);

	cudaFree(d_param);
	free(h_param);
	return d_scene;
}

Scene* h_initScene(uint32_t* h_texture, int t){
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

	Scene *scene = new Scene(9, h_texture);
	scene->addLight(-1,8,6,10);
	scene->setHorizonColour(black);
	scene->addPlane(v1,v2,v3,v4,bright_green,SHINY);
	scene->addTri(v3,v4,v5,bright_green,SHINY);
	scene->addSphere(2,10,5,2.5,dark_red,SHINY);
	scene->addSphere(6,9,3,t,cold_blue,DIFFUSE);
	scene->addSphere(6,7,-1,2,cold_blue,SHINY);
	scene->addSphere(-2,6,0,1.2f,soft_red,WATER);
	scene->addSphere(-6,8,-2,2,soft_red,GLASS);
	scene->addSphere(-9,8,3,3,bright_green,SHINY);

	//auto parser

	scenePrototype_t exterior = parseFile();
	vertex_t* verts = exterior.verts;
	triPrototype_t* tris = exterior.tris;
	int numOfTris = exterior.numOfTris;

	for(int i=0; i<numOfTris; i++){
		//1 indexed so must switch to 0 indexed
		scene->addTri(verts[tris[i].v1-1], verts[tris[i].v2-1], verts[tris[i].v3-1], cold_blue, DIFFUSE);
	}

	//auto parser

	return scene;
}

__global__ void cudaShootRays(vector_t* lightRay, colour_t* colGrid, Scene* d_scene){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	Ray ray(lightRay[index], d_scene, MAX_ITERATIONS);
	colGrid[index] = ray.raytrace();
}

void cpuShootRays(vector_t* lightRay, colour_t* colGrid, Scene* h_scene, int numOfRays){
	for(int i=0;i<numOfRays;i++){
		Ray ray(lightRay[i], h_scene, MAX_ITERATIONS);
		colGrid[i] = ray.raytrace();
	}
}

void d_destroyScene(Scene* d_scene, uint32_t* d_textureData){
	cudaFree(d_textureData);
	cudaFree(d_scene);
}

void h_destroyScene(Scene* h_scene){
	free(h_scene);
}

//where x and y are the top left most coordinates and squareSize is one block being rendered
void drawPixelRaytracer(SDL_Renderer *renderer, int x, int y, int squareSizeX, int squareSizeY, Scene* scene){

	//vector_t locDir = scene->getCamera().getLocDir();
	vector_t locDir(0,0,0,0,3,0);
	//float ZOOM_FACTOR = scene->getCamera().getGridSize();
	float ZOOM_FACTOR = 0.01f;

	
	vector_t* thisLocDir = (vector_t*)malloc(squareSizeX*squareSizeY*sizeof(vector_t));
	colour_t* col = (colour_t*)malloc(squareSizeX*squareSizeY*sizeof(colour_t));

	for(int i=x*squareSizeX;i<(x+1)*squareSizeX;i++){
		for(int j=y*squareSizeY;j<(y+1)*squareSizeY;j++){
			int index = (j-y*squareSizeY)*squareSizeX+(i-x*squareSizeX);
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



	if(USE_CUDA){

		vector_t* d_lightRay;
		colour_t* d_colourGrid;

		cudaMalloc((void**) &d_lightRay, sizeof(vector_t)*squareSizeX*squareSizeY);
		cudaMalloc((void**) &d_colourGrid, sizeof(colour_t)*squareSizeX*squareSizeY);

		cudaMemcpy(d_lightRay, thisLocDir, sizeof(vector_t)*squareSizeX*squareSizeY, cudaMemcpyHostToDevice);

		
		cudaShootRays<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(d_lightRay, d_colourGrid, scene);

		cudaMemcpy(col, d_colourGrid, sizeof(colour_t)*squareSizeX*squareSizeY, cudaMemcpyDeviceToHost);


		cudaFree(d_lightRay);
		cudaFree(d_colourGrid);
	}else{
		cpuShootRays(thisLocDir, col, scene, NUM_OF_BLOCKS*THREADS_PER_BLOCK);
	}


	for(int i=x*squareSizeX;i<(x+1)*squareSizeX;i++){
		for(int j=y*squareSizeY;j<(y+1)*squareSizeY;j++){
			int index = (j-y*squareSizeY)*squareSizeX+(i-x*squareSizeX);

			if(col[index].r<=255 && col[index].g<=255 && col[index].b<=255){
				SDL_SetRenderDrawColor(renderer, (int)col[index].r, (int)col[index].g, (int)col[index].b, 255);
			}else{
				//handle overspill colours
				processColourOverspill(renderer, col[index]);
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
	Scene* scene;
	for(int t=0; t<MAX_ANIMATION_ITERATIONS; t++){
		if(USE_CUDA){
			scene = d_initScene(tData, t);
		}else{
			scene = h_initScene(tData, t);
		}

		if(USE_BLOCK_BY_BLOCKING_RENDERING){
			for(int j=0; j<SCREEN_HEIGHT/RENDER_SQUARE_SIZE; j++){
				for(int i=0; i<SCREEN_WIDTH/RENDER_SQUARE_SIZE; i++){
					//CALL OUR DRAW LOOP FUNCTION
					drawPixelRaytracer(renderer, i, j, RENDER_SQUARE_SIZE, RENDER_SQUARE_SIZE, scene);
					SDL_RenderPresent(renderer);
				}
			}
		}else{
			drawPixelRaytracer(renderer, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, scene);
			SDL_RenderPresent(renderer);
		}
	}
	


	printf("done");
	if(USE_CUDA){
		d_destroyScene(scene, tData);
	}else{
		h_destroyScene(scene);
	}
	IMG_Quit();
	free(tData);
	SDL_Delay(DISPLAY_TIME);
	//Destroy window
    SDL_DestroyWindow(window);
    //Quit SDL subsystems
    SDL_Quit();
    return 0;
}

//UTILITY FUNCTION TO SCALE COLOURS
void processColourOverspill(SDL_Renderer *renderer, colour_t col){
	float max;
	int r = col.r;
	int g = col.g;
	int b = col.b;
	if(r>b && g>b){
		max = (float)b;
	}else if(r>g){
		max = (float)r;
	}else{
		max = (float)g;
	}
	float multiplier = 255/max;
	SDL_SetRenderDrawColor(renderer, (int)r*multiplier, (int)g*multiplier, (int)b*multiplier, 255);
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
    case 2:
        return *(uint16_t *)p;
    case 3:
        if(SDL_BYTEORDER == SDL_BIG_ENDIAN)
            return p[0] << 16 | p[1] << 8 | p[2];
        else
            return p[0] | p[1] << 8 | p[2] << 16;
    case 4:
        return *(uint32_t *)p;
    default:
        return 0;       /* shouldn't happen, but avoids warnings */
    }
}