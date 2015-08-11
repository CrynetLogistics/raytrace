#include <iostream>
#include "SDL.h"
#include "stdio.h"
#include "math.h"
#include "Scene.h"
#include "structures.h"
#include "vector_t.h"
#include "Plane.h"

#undef main
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
#define DISPLAY_TIME 10000
#define BRIGHTNESS 15
#define CLIPPING_DISTANCE 999

void drawPixelRaytracer(SDL_Renderer *renderer, Scene *scene);

int main()
{
    SDL_Window* window = NULL;
    SDL_Surface* screenSurface = NULL;
	SDL_Init(SDL_INIT_EVERYTHING);

	//create window
	window = SDL_CreateWindow("Mandelbrot", 
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    
	SDL_Renderer *renderer = NULL;
	renderer = SDL_CreateRenderer(window, 0, SDL_RENDERER_ACCELERATED);
	//BACKGROUND COLOUR SET
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
	SDL_RenderClear(renderer);



	colour_t col1;
	col1.r = 150;
	col1.g = 0;
	col1.b = 0;
	colour_t col2;
	col2.r = 255;
	col2.g = 77;
	col2.b = 99;
	colour_t col3;
	col3.r = 15;
	col3.g = 240;
	col3.b = 88;




	

	Scene scene;

	scene.addSphere(2,10,5,2,col1);
	scene.addSphere(0,12,0,4,col2);
	scene.addSphere(-9,8,3,2,col3);

	//CALL OUR DRAW LOOP FUNCTION
	drawPixelRaytracer(renderer, &scene);



	SDL_RenderPresent(renderer);
	SDL_Delay(DISPLAY_TIME);
	//Destroy window
    SDL_DestroyWindow(window);
    //Quit SDL subsystems
    SDL_Quit();
    return 0;
}

colour_t calculateIntensityFromIntersections(vector_t lightRay, Scene *scene){
	//find nicer way of starting tMin
	float tMin = CLIPPING_DISTANCE;
	float tCurrent = 0;
	int iMin = 0;

	float currentR = 0;
	float currentG = 0;
	float currentB = 0;
	for(int i=0;i<scene->getNumOfSpheres();i++){
		//getSphere will become getGENERAL with GENERAL encompassing spheres, planes etc
		tCurrent = scene->getSphere(i).getIntersectionParameter(lightRay, scene->getLight());
		if(tCurrent<tMin && tCurrent!=0){
			tMin = tCurrent;
			iMin = i;
		}
	}

	if(tMin!=CLIPPING_DISTANCE){
		currentR = (float)scene->getSphere(iMin).getColour().r;
		currentG = (float)scene->getSphere(iMin).getColour().g;
		currentB = (float)scene->getSphere(iMin).getColour().b;
	}

	//testing will generalise later
	tCurrent = scene->getPlane(iMin).getIntersectionParameter(lightRay, scene->getLight());
	if(tCurrent<tMin && tCurrent!=0){
		tMin = tCurrent;
		currentR=(float)scene->getPlane(0).getColour().r;
		currentG=(float)scene->getPlane(0).getColour().g;
		currentB=(float)scene->getPlane(0).getColour().b;
	}

	

	bool isShadowed = scene->getSphere(iMin).getShadowedStatus(lightRay, tMin, scene->getLight());

	float distance = lightRay.calculateDistance(tMin);

	colour_t currentColour;
	if(!isShadowed){
		currentColour.r = (int)(currentR*distance*BRIGHTNESS)/255;
		currentColour.g = (int)(currentG*distance*BRIGHTNESS)/255;
		currentColour.b = (int)(currentB*distance*BRIGHTNESS)/255;
	}else{
		currentColour.r = (int)(currentR*distance*BRIGHTNESS/2)/255;
		currentColour.g = (int)(currentG*distance*BRIGHTNESS/2)/255;
		currentColour.b = (int)(currentB*distance*BRIGHTNESS/2)/255;
	}
	return currentColour;
}



void drawPixelRaytracer(SDL_Renderer *renderer, Scene *scene){
	SDL_Rect r;
	r.h = 1;
	r.w = 1;

	vector_t locDir = scene->getCamera().getLocDir();
	float ZOOM_FACTOR = scene->getCamera().getGridSize();

	for(int i=0;i<SCREEN_WIDTH;i++){
		for(int j=0;j<SCREEN_HEIGHT;j++){
			r.x = i;
			r.y = j;

			//TODO: GENERALISE THIS FORMULA
			//CURRENTLY: ONLY APPLIES TO CAMERA POINTING ALONG Y DIRECTION
			vector_t thisLocDir;
			thisLocDir.x0 = locDir.x0;
			thisLocDir.y0 = locDir.y0;
			thisLocDir.z0 = locDir.z0;

			thisLocDir.xt = locDir.xt + (float)(i-SCREEN_WIDTH/2)*ZOOM_FACTOR;
			thisLocDir.yt = locDir.yt;
			thisLocDir.zt = locDir.zt + (float)(SCREEN_HEIGHT/2-j)*ZOOM_FACTOR;

			colour_t col = calculateIntensityFromIntersections(thisLocDir, scene);
			
			if(col.r<=255 && col.g<=255 && col.b<=255){
				SDL_SetRenderDrawColor(renderer, col.r, col.g, col.b, 255);
			}else{
				SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
			}
			
			SDL_RenderFillRect(renderer, &r);
		}
	}
}