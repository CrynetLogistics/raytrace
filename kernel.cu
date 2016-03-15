#include <iostream>
#include <string>
#include <ctime>
#include "linux/sdl_src/SDL2/include/SDL.h"
#include "linux/sdl_src/SDL2_image/SDL_image.h"
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
#include "kernel.h"

#undef main
//-----------------------------------------------------------------------------

//IF USE_BLOCK_BY_BLOCKING_RENDERING == 1
//	THEN RENDER_SQUARE_SIZE*RENDER_SQUARE_SIZE = THREADS_PER_BLOCK*NUM_OF_BLOCKS
//	ELSE SCREEN_WIDTH*SCREEN_HEIGHT = THREADS_PER_BLOCK*NUM_OF_BLOCKS
#define RENDER_SQUARE_SIZE 80
//
#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 25
//

std::string FILENAME;
int USE_CUDA = 0;
int SCREEN_WIDTH = 1280;
int SCREEN_HEIGHT = 720;
int MSAA_LEVEL = 0;
int BSPBVH_DEPTH = 0;
int ENABLE_TEXTURES = 1;
int DEBUG_LEVEL = 0;

#define USE_BLOCK_BY_BLOCKING_RENDERING 1

//-----------------------------------------------------------------------------
#define MAX_ANIMATION_ITERATIONS 1 // 1 for just a still image
#define DISPLAY_TIME 300
#define MAX_ITERATIONS 3
#define TEXTURE_WIDTH 600
#define TEXTURE_HEIGHT 300
//-----------------------------------------------------------------------------
uint32_t getpixel(SDL_Surface *surface, int x, int y);
void processColourOverspill(SDL_Surface *rendererAux, colour_t col, int i, int j);
bool saveScreenshotBMP(std::string filepath, SDL_Window* SDLWindow, SDL_Renderer* SDLRenderer);
//-----------------------------------------------------------------------------

__global__ void d_initScene(Scene* d_scene, uint32_t* textureData, int* d_param,
							int* d_numOfTris, vertex_t* d_verts, triPrototype_t* d_tris){
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

	v1.x = -30;v1.y = 0;v1.z = -3;
	v2.x = 30;v2.y = 0;v2.z = -3;
	v3.x = -30;v3.y = 50;v3.z = -3;
	v4.x = 30;v4.y = 50;v4.z = -3;
	v5.x = -30;v5.y = 50;v5.z = 30;
	v6.x = 30;v6.y = 50;v6.z = 30;

	//6 Meshes; Meshes = {Spheres, Planes}
	d_scene = new (d_scene) Scene(4 + *d_numOfTris, textureData);
	d_scene->addLight(-1,8,6,10);
	d_scene->setHorizonColour(black);
	d_scene->addTri(v1,v2,v3,bright_green, SHINY);
	d_scene->addTri(v2,v3,v4,bright_green, SHINY);
	d_scene->addTri(v3,v4,v5,bright_green, SHINY);
	d_scene->addTri(v4,v5,v6,bright_green, SHINY);

	//auto parser

	for(int i=0; i<*d_numOfTris; i++){
		d_scene->addTri(d_verts[d_tris[i].v1-1], d_verts[d_tris[i].v2-1], d_verts[d_tris[i].v3-1], cold_blue, DIFFUSE);
	}
}

__global__ void d_buildBSPBVH(Scene* d_scene, int* d_buildBSPBVH, Stack<BinTreeNode*> *d_unPropagatedNodes){
	int bottomMax = powf(2.0f, (float) *d_buildBSPBVH);
	d_unPropagatedNodes = new (d_unPropagatedNodes) Stack<BinTreeNode*>(bottomMax);

	d_scene->buildBSPBVH(*d_buildBSPBVH, d_unPropagatedNodes);
}

__global__ void d_continuePropagation(Stack<BinTreeNode*> *d_unPropagatedNodes){
	d_unPropagatedNodes->pop()->propagateTree(d_unPropagatedNodes);
}

Scene* d_initScene(uint32_t* h_texture, int t){
	Scene* d_scene;
	uint32_t* d_textureData;
	int* h_param = (int*)malloc(sizeof(int));
	*h_param = t;
	int* d_param;


	//auto parser

	scenePrototype_t exterior = parseFile(FILENAME, DEBUG_LEVEL);
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
	cudaMalloc((void**) &d_scene, sizeof(Scene));
	
	cudaMemcpy(d_param, h_param, sizeof(int), cudaMemcpyHostToDevice);

	if(ENABLE_TEXTURES){
		cudaMalloc((void**) &d_textureData, sizeof(uint32_t)*TEXTURE_HEIGHT*TEXTURE_WIDTH);
		cudaMemcpy(d_textureData, h_texture, sizeof(uint32_t)*TEXTURE_HEIGHT*TEXTURE_WIDTH, cudaMemcpyHostToDevice);
	}else{
		cudaMalloc((void**) &d_textureData, 1);
	}

	d_initScene<<<1,1>>>(d_scene, d_textureData, d_param, d_numOfTris, d_verts, d_tris);

	if(BSPBVH_DEPTH!=0){
		std::cout<<"Building BSP BVH on GPU...";
		int* d_BSPBVH_DEPTH;
		Stack<BinTreeNode*> *d_unPropagatedNodes;
		Stack<BinTreeNode*> *h_unPropagatedNodes = (Stack<BinTreeNode*> *)malloc(sizeof(Stack<BinTreeNode>));

		cudaMalloc((void**) &d_unPropagatedNodes, sizeof(Stack<BinTreeNode*>));
		cudaMalloc((void**) &d_BSPBVH_DEPTH, sizeof(int));
		cudaMemcpy(d_BSPBVH_DEPTH, &BSPBVH_DEPTH, sizeof(int), cudaMemcpyHostToDevice);

		d_buildBSPBVH<<<1,1>>>(d_scene, d_BSPBVH_DEPTH, d_unPropagatedNodes);

		cudaMemcpy(h_unPropagatedNodes, d_unPropagatedNodes, sizeof(Stack<BinTreeNode*>), cudaMemcpyDeviceToHost);

		while(!h_unPropagatedNodes->isEmpty()){
		
			d_continuePropagation<<<1,1>>>(d_unPropagatedNodes);
			cudaMemcpy(h_unPropagatedNodes, d_unPropagatedNodes, sizeof(Stack<BinTreeNode*>), cudaMemcpyDeviceToHost);
		}

		cudaFree(d_BSPBVH_DEPTH);
		cudaFree(d_unPropagatedNodes);
		free(h_unPropagatedNodes);
		std::cout<<"done"<<std::endl;
	}
	

	cudaFree(d_param);
	free(h_param);
	return d_scene;
}

Scene* h_initScene(uint32_t* h_texture, int t){
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

	v1.x = -30;v1.y = 0;v1.z = -3;
	v2.x = 30;v2.y = 0;v2.z = -3;
	v3.x = -30;v3.y = 50;v3.z = -3;
	v4.x = 30;v4.y = 50;v4.z = -3;
	v5.x = -30;v5.y = 50;v5.z = 30;
	v6.x = 30;v6.y = 50;v6.z = 30;

	//autoparser
	scenePrototype_t exterior = parseFile(FILENAME, DEBUG_LEVEL);

	Scene *scene = new Scene(4 + exterior.numOfTris, h_texture);
	scene->addLight(-1,8,6,10);
	scene->setHorizonColour(black);
	scene->addTri(v1,v2,v3,bright_green, SHINY);
	scene->addTri(v2,v3,v4,bright_green, SHINY);
	scene->addTri(v3,v4,v5,bright_green, SHINY);
	scene->addTri(v4,v5,v6,bright_green, SHINY);

	//auto parser

	vertex_t* verts = exterior.verts;
	triPrototype_t* tris = exterior.tris;
	int numOfTris = exterior.numOfTris;

	for(int i=0; i<numOfTris; i++){
		//1 indexed so must switch to 0 indexed
		scene->addTri(verts[tris[i].v1-1], verts[tris[i].v2-1], verts[tris[i].v3-1], cold_blue, DIFFUSE);
	}

	//auto parser

	
	if(BSPBVH_DEPTH!=0){
		std::cout<<"Building BSP BVH on CPU...";
		scene->buildBSPBVH(BSPBVH_DEPTH);
		std::cout<<"done"<<std::endl;
	}

	return scene;
}

__global__ void cudaShootRays(launchParams_t* thisLaunch, colour_t* colGrid, Scene* d_scene){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int xPosi = index%thisLaunch->squareSizeX + thisLaunch->x*thisLaunch->squareSizeX;
	int yPosj = index/thisLaunch->squareSizeX + thisLaunch->y*thisLaunch->squareSizeY;

	vector_t init_vector = d_scene->getCamera().getThisLocationDirection(
		xPosi, yPosj, thisLaunch->SCREEN_X, thisLaunch->SCREEN_Y, thisLaunch->MSAA_SAMPLES, thisLaunch->MSAA_INDEX);

	Ray ray(init_vector, d_scene, MAX_ITERATIONS, thisLaunch->BSPBVH_DEPTH);
	colGrid[index] = ray.raytrace();
}

void cpuShootRays(colour_t* colGrid, Scene* h_scene, int numOfRays, launchParams_t* thisLaunch){
	for(int i=0;i<numOfRays;i++){
		int xPosi = i%thisLaunch->squareSizeX + thisLaunch->x*thisLaunch->squareSizeX;
		int yPosj = i/thisLaunch->squareSizeX + thisLaunch->y*thisLaunch->squareSizeY;

		vector_t init_vector = h_scene->getCamera().getThisLocationDirection(
			xPosi, yPosj, SCREEN_WIDTH, SCREEN_HEIGHT, thisLaunch->MSAA_SAMPLES, thisLaunch->MSAA_INDEX);

		Ray ray(init_vector, h_scene, MAX_ITERATIONS, thisLaunch->BSPBVH_DEPTH);
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
void drawPixelRaytracer(SDL_Renderer *renderer, launchParams_t* thisLaunch, Scene* scene, SDL_Surface* rendererAux){
	int samplesToRender = pow(MSAA_LEVEL+1, 2);
	int numberOfPixels = thisLaunch->squareSizeX*thisLaunch->squareSizeY;

	//a list of different renderings of the scene
	colour_t** colDeck = (colour_t**)malloc(sizeof(colour_t*)*samplesToRender);

	for(int i=0; i<samplesToRender; i++){
		colour_t* col = (colour_t*)malloc(numberOfPixels*sizeof(colour_t));

		thisLaunch->MSAA_INDEX = i;
		if(USE_CUDA){
			colour_t* d_colourGrid;
			launchParams_t* d_thisLaunch;

			cudaMalloc((void**) &d_thisLaunch, sizeof(launchParams_t));
			cudaMalloc((void**) &d_colourGrid, sizeof(colour_t)*thisLaunch->squareSizeX*thisLaunch->squareSizeY);

			cudaMemcpy(d_thisLaunch, thisLaunch, sizeof(launchParams_t), cudaMemcpyHostToDevice);

			cudaShootRays<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(d_thisLaunch, d_colourGrid, scene);

			cudaMemcpy(col, d_colourGrid, sizeof(colour_t)*thisLaunch->squareSizeX*thisLaunch->squareSizeY, cudaMemcpyDeviceToHost);


			cudaFree(d_thisLaunch);
			cudaFree(d_colourGrid);
		}else{
			cpuShootRays(col, scene, NUM_OF_BLOCKS*THREADS_PER_BLOCK, thisLaunch);
		}
		colDeck[i] = col;
	}

	colour_t* finalColour = (colour_t*)calloc(numberOfPixels, sizeof(colour_t));

	for(int i=0; i<numberOfPixels; i++){
		for(int j=0; j<samplesToRender; j++){
			finalColour[i].r += colDeck[j][i].r/samplesToRender;
			finalColour[i].g += colDeck[j][i].g/samplesToRender;
			finalColour[i].b += colDeck[j][i].b/samplesToRender;
		}
	}

	for(int i=thisLaunch->x*thisLaunch->squareSizeX;i<(thisLaunch->x+1)*thisLaunch->squareSizeX;i++){
		for(int j=thisLaunch->y*thisLaunch->squareSizeY;j<(thisLaunch->y+1)*thisLaunch->squareSizeY;j++){
			int index = (j-thisLaunch->y*thisLaunch->squareSizeY)*thisLaunch->squareSizeX+(i-thisLaunch->x*thisLaunch->squareSizeX);

			if(finalColour[index].r<=255 && finalColour[index].g<=255 && finalColour[index].b<=255){
				//SDL_SetRenderDrawColor(renderer, (int)finalColour[index].r, (int)finalColour[index].g, (int)finalColour[index].b, 255);
                SDL_Rect r;
                r.x = i;
                r.y = j;
                r.w = 1;
                r.h = 1;
                
                SDL_FillRect(rendererAux, &r, SDL_MapRGB(rendererAux->format, (int)finalColour[index].r, (int)finalColour[index].g, (int)finalColour[index].b));
			}else{
				//handle overspill colours
				processColourOverspill(rendererAux, finalColour[index], i, j);
			}
			//SDL_RenderDrawPoint(renderer, i, j);
		}
	}

	for(int i=0; i<samplesToRender; i++){
		free(colDeck[i]);
	}
	free(colDeck);
	free(finalColour);
}

int raytrace(int USE_GPU_i, int SCREEN_WIDTH_i, int SCREEN_HEIGHT_i, std::string FILENAME_i,
			 int MSAA_LEVEL_i, int BSPBVH_DEPTH_i, int ENABLE_TEXTURES_i, int DEBUG_LEVEL_i)
{
	std::clock_t start;
	start = std::clock();
	double afterSceneCreation;
	double beforeSceneCreation;

    SDL_Surface *rendererAux = SDL_CreateRGBSurface(0, SCREEN_WIDTH, SCREEN_HEIGHT, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);

	///
	MSAA_LEVEL = MSAA_LEVEL_i;
	USE_CUDA = USE_GPU_i;
	FILENAME = FILENAME_i;
	SCREEN_HEIGHT = SCREEN_HEIGHT_i;
	SCREEN_WIDTH = SCREEN_WIDTH_i;
	BSPBVH_DEPTH = BSPBVH_DEPTH_i;
	ENABLE_TEXTURES = ENABLE_TEXTURES_i;
	DEBUG_LEVEL = DEBUG_LEVEL_i;

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
	uint32_t* tData;
	if(ENABLE_TEXTURES){
		texture = IMG_Load("texture.png");
		if(!texture){
			printf("ERROR:%s", IMG_GetError());
		}


		tData = (uint32_t*)malloc(TEXTURE_WIDTH*TEXTURE_HEIGHT*sizeof(uint32_t));
		for(int i=0;i<TEXTURE_WIDTH*TEXTURE_HEIGHT;i++){
			tData[i] = getpixel(texture, i%TEXTURE_WIDTH, i/TEXTURE_WIDTH);
		}
	}else{
		uint32_t tDataEl = 0;
		tData = &tDataEl;
	}
	//END OF TEXTURE LOAD
	Scene* scene;

	launchParams_t* thisLaunch = (launchParams_t*)malloc(sizeof(launchParams_t));
	thisLaunch->SCREEN_X = SCREEN_WIDTH;
	thisLaunch->SCREEN_Y = SCREEN_HEIGHT;
	thisLaunch->MSAA_SAMPLES = MSAA_LEVEL+1;
	thisLaunch->BSPBVH_DEPTH = BSPBVH_DEPTH_i;
	for(int t=0; t<MAX_ANIMATION_ITERATIONS; t++){

		//takes time signature before scene is built
		beforeSceneCreation = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		if(USE_CUDA){
			scene = d_initScene(tData, t);
		}else{
			scene = h_initScene(tData, t);
		}

		//takes time signature after scene is built
		afterSceneCreation = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				
		if(USE_BLOCK_BY_BLOCKING_RENDERING){
			for(int j=0; j<SCREEN_HEIGHT/RENDER_SQUARE_SIZE; j++){
				for(int i=0; i<SCREEN_WIDTH/RENDER_SQUARE_SIZE; i++){
					//CALL OUR DRAW LOOP FUNCTION
					thisLaunch->squareSizeX = RENDER_SQUARE_SIZE;
					thisLaunch->squareSizeY = RENDER_SQUARE_SIZE;
					thisLaunch->x = i;
					thisLaunch->y = j;
					
					drawPixelRaytracer(renderer, thisLaunch, scene, rendererAux);
					SDL_RenderPresent(renderer);
				}
			}
		}else{
			thisLaunch->squareSizeX = SCREEN_WIDTH;
			thisLaunch->squareSizeY = SCREEN_HEIGHT;
			thisLaunch->x = 0;
			thisLaunch->y = 0;
			drawPixelRaytracer(renderer, thisLaunch, scene, rendererAux);
			SDL_RenderPresent(renderer);
		}
	}




    //http://stackoverflow.com/questions/22315980/sdl2-c-taking-a-screenshot
    //SDL_Surface *sshot = SDL_CreateRGBSurface(0, SCREEN_WIDTH, SCREEN_HEIGHT, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
    //SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_ARGB8888, sshot->pixels, sshot->pitch);
	//std::string outName = FILENAME_i;
	//outName.append("_out.bmp");
	//SDL_SaveBMP(sshot, outName.c_str());
    //SDL_FreeSurface(sshot);


    std::string outName2 = FILENAME_i;
	outName2.append("_out2.bmp");
	SDL_SaveBMP(rendererAux, outName2.c_str());
    SDL_FreeSurface(rendererAux);
	




	double timeDuration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	printf("Done in %.3fs\n\twith scene creation taking %.3fs\n", timeDuration, afterSceneCreation - beforeSceneCreation);
	if(USE_CUDA){
		d_destroyScene(scene, tData);
	}else{
		h_destroyScene(scene);
	}
	IMG_Quit();
	if(ENABLE_TEXTURES){
		free(tData);
	}
	free(thisLaunch);
	SDL_Delay(DISPLAY_TIME);

	//Destroy window
    SDL_DestroyWindow(window);
    //Quit SDL subsystems
    SDL_Quit();
    return 0;
}

//UTILITY FUNCTION TO SCALE COLOURS
void processColourOverspill(SDL_Surface *rendererAux, colour_t col, int i, int j){
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

    SDL_Rect rr;
    rr.x = i;
    rr.y = j;
    rr.w = 1;
    rr.h = 1;

    SDL_FillRect(rendererAux, &rr, SDL_MapRGB(rendererAux->format, (int)r*multiplier, (int)g*multiplier, (int)b*multiplier));
//	SDL_SetRenderDrawColor(renderer, (int)r*multiplier, (int)g*multiplier, (int)b*multiplier, 255);
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
