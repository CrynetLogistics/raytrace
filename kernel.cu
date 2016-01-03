#include <iostream>
#include <string>
#include <ctime>
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
//#define THREADS_PER_BLOCK 1024
//#define NUM_OF_BLOCKS 900
//

std::string FILENAME;
int USE_CUDA = 0;
int SCREEN_WIDTH = 1280;
int SCREEN_HEIGHT = 720;
int MSAA_LEVEL = 0;
int BSPBVH_DEPTH = 0;

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
bool saveScreenshotBMP(std::string filepath, SDL_Window* SDLWindow, SDL_Renderer* SDLRenderer);
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
	d_scene = new (d_scene) Scene(4 + *d_numOfTris, textureData);
	//d_scene = new Scene(9, textureData);
	d_scene->addLight(-1,8,6,10);
	d_scene->setHorizonColour(black);
	//d_scene->addPlane(v1,v2,v3,v4,bright_green,SHINY);
	//scene->addPlane(v3,v4,v5,v6,bright_green,SHINY);
	//d_scene->addPlane(v3,v4,v5,v6,bright_green,SHINY);
	//scene->addPlane(v7,v8,v5,v6,bright_green,DIFFUSE);
	d_scene->addTri(v1,v2,v3,bright_green, SHINY);
	d_scene->addTri(v2,v3,v4,bright_green, SHINY);
	d_scene->addTri(v3,v4,v5,bright_green, SHINY);
	d_scene->addTri(v4,v5,v6,bright_green, SHINY);
	//scene->addPlane(v1,v3,v5,v7,bright_green,DIFFUSE);
	//scene->addPlane(v2,v4,v6,v8,bright_green,DIFFUSE);
	/*d_scene->addSphere(2,10,5,2.5,dark_red,SHINY);
	d_scene->addSphere(6,9,3,3,cold_blue,DIFFUSE);
	d_scene->addSphere(6,7,-1,2,cold_blue,SHINY);
	d_scene->addSphere(-2,6,0,1.2f,soft_red,WATER);
	d_scene->addSphere(*d_param-6,8,-2,2,soft_red,GLASS);
	d_scene->addSphere(-9,8,3,3,bright_green,SHINY);*/

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

	scenePrototype_t exterior = parseFile(FILENAME);
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

	if(BSPBVH_DEPTH!=0){
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
	}

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

	//autoparser
	scenePrototype_t exterior = parseFile(FILENAME);

	Scene *scene = new Scene(4 + exterior.numOfTris, h_texture);
	scene->addLight(-1,8,6,10);
	scene->setHorizonColour(black);
	//scene->addPlane(v1,v2,v3,v4,bright_green,SHINY);
	//scene->addPlane(v3,v4,v5,v6,bright_green,SHINY);
	scene->addTri(v1,v2,v3,bright_green, SHINY);
	scene->addTri(v2,v3,v4,bright_green, SHINY);
	scene->addTri(v3,v4,v5,bright_green, SHINY);
	scene->addTri(v4,v5,v6,bright_green, SHINY);

	//scene->addSphere(2,10,5,2.5,dark_red,SHINY);
	//scene->addSphere(6,9,3,t,cold_blue,DIFFUSE);
	//scene->addSphere(6,7,-1,2,cold_blue,SHINY);
	//scene->addSphere(-2,6,0,1.2f,soft_red,WATER);
	//scene->addSphere(-6,8,-2,2,soft_red,GLASS);
	//scene->addSphere(-9,8,3,3,bright_green,SHINY);

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
		scene->buildBSPBVH(BSPBVH_DEPTH);
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
void drawPixelRaytracer(SDL_Renderer *renderer, launchParams_t* thisLaunch, Scene* scene){
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
				SDL_SetRenderDrawColor(renderer, (int)finalColour[index].r, (int)finalColour[index].g, (int)finalColour[index].b, 255);
			}else{
				//handle overspill colours
				processColourOverspill(renderer, finalColour[index]);
			}
			SDL_RenderDrawPoint(renderer, i, j);
		}
	}

	for(int i=0; i<samplesToRender; i++){
		free(colDeck[i]);
	}
	free(colDeck);
	free(finalColour);
}

int raytrace(int USE_GPU_i, int SCREEN_WIDTH_i, int SCREEN_HEIGHT_i, std::string FILENAME_i, int MSAA_LEVEL_i, int BSPBVH_DEPTH_i)
{
	std::clock_t start;
	start = std::clock();

	///
	MSAA_LEVEL = MSAA_LEVEL_i;
	USE_CUDA = USE_GPU_i;
	FILENAME = FILENAME_i;
	SCREEN_HEIGHT = SCREEN_HEIGHT_i;
	SCREEN_WIDTH = SCREEN_WIDTH_i;
	BSPBVH_DEPTH = BSPBVH_DEPTH_i;

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

	launchParams_t* thisLaunch = (launchParams_t*)malloc(sizeof(launchParams_t));
	thisLaunch->SCREEN_X = SCREEN_WIDTH;
	thisLaunch->SCREEN_Y = SCREEN_HEIGHT;
	thisLaunch->MSAA_SAMPLES = MSAA_LEVEL+1;
	thisLaunch->BSPBVH_DEPTH = BSPBVH_DEPTH_i;
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
					thisLaunch->squareSizeX = RENDER_SQUARE_SIZE;
					thisLaunch->squareSizeY = RENDER_SQUARE_SIZE;
					thisLaunch->x = i;
					thisLaunch->y = j;
					
					drawPixelRaytracer(renderer, thisLaunch, scene);
					SDL_RenderPresent(renderer);
				}
			}
		}else{
			thisLaunch->squareSizeX = SCREEN_WIDTH;
			thisLaunch->squareSizeY = SCREEN_HEIGHT;
			thisLaunch->x = 0;
			thisLaunch->y = 0;
			drawPixelRaytracer(renderer, thisLaunch, scene);
			SDL_RenderPresent(renderer);
		}
	}
	saveScreenshotBMP("world.bmp", window, renderer);
	


	double timeDuration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	printf("done in %.3fs",timeDuration);
	if(USE_CUDA){
		d_destroyScene(scene, tData);
	}else{
		h_destroyScene(scene);
	}
	IMG_Quit();
	free(tData);
	free(thisLaunch);
	while(true){
		SDL_Delay(DISPLAY_TIME);
	}
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

//UTILITY FUNCTION COURTESY OF stackoverflow.com/questions/20233469/how-do-i-take-and-save-a-bmp-screenshot-in-sdl-2
bool saveScreenshotBMP(std::string filepath, SDL_Window* SDLWindow, SDL_Renderer* SDLRenderer) {
    SDL_Surface* saveSurface = NULL;
    SDL_Surface* infoSurface = NULL;
    infoSurface = SDL_GetWindowSurface(SDLWindow);
    if (infoSurface == NULL) {
        std::cerr << "Failed to create info surface from window in saveScreenshotBMP(string), SDL_GetError() - " << SDL_GetError() << "\n";
    } else {
        unsigned char * pixels = new (std::nothrow) unsigned char[infoSurface->w * infoSurface->h * infoSurface->format->BytesPerPixel];
        if (pixels == 0) {
            std::cerr << "Unable to allocate memory for screenshot pixel data buffer!\n";
            return false;
        } else {
            if (SDL_RenderReadPixels(SDLRenderer, &infoSurface->clip_rect, infoSurface->format->format, pixels, infoSurface->w * infoSurface->format->BytesPerPixel) != 0) {
                std::cerr << "Failed to read pixel data from SDL_Renderer object. SDL_GetError() - " << SDL_GetError() << "\n";
                pixels = NULL;
                return false;
            } else {
                saveSurface = SDL_CreateRGBSurfaceFrom(pixels, infoSurface->w, infoSurface->h, infoSurface->format->BitsPerPixel, infoSurface->w * infoSurface->format->BytesPerPixel, infoSurface->format->Rmask, infoSurface->format->Gmask, infoSurface->format->Bmask, infoSurface->format->Amask);
                if (saveSurface == NULL) {
                    std::cerr << "Couldn't create SDL_Surface from renderer pixel data. SDL_GetError() - " << SDL_GetError() << "\n";
                    return false;
                }
                SDL_SaveBMP(saveSurface, filepath.c_str());
                SDL_FreeSurface(saveSurface);
                saveSurface = NULL;
            }
            delete[] pixels;
        }
        SDL_FreeSurface(infoSurface);
        infoSurface = NULL;
    }
    return true;
}