#include <iostream>
#include "SDL.h"
#include "stdio.h"
#include "math.h"
#include "Scene.h"
#include "structures.h"
#include "vector_t.h"
#include "Plane.h"
#include "Ray.h"

#undef main
#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
#define DISPLAY_TIME 30000

void drawPixelRaytracer(SDL_Renderer *renderer, Scene *scene);

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
	colour_t col4;
	col4.r = 12;
	col4.g = 37;
	col4.b = 255;


	vertex_t v1;
	vertex_t v2;
	vertex_t v3;
	vertex_t v4;
	v1.x = -30;v1.y = 0;v1.z = -3;
	v2.x = 30;v2.y = 0;v2.z = -3;
	v3.x = -30;v3.y = 50;v3.z = -3;
	v4.x = 30;v4.y = 50;v4.z = -3;

	vertex_t v5;
	vertex_t v6;
	vertex_t v7;
	vertex_t v8;
	v5.x = -30;v5.y = 50;v5.z = -3;
	v6.x = 30;v6.y = 50;v6.z = -3;
	v7.x = -30;v7.y = 50;v7.z = 50;
	v8.x = 30;v8.y = 50;v8.z = 50;


	Scene scene;
	scene.addLight(-1,8,6,10);
	scene.addPlane(v1,v2,v3,v4,col3);
	scene.addPlane(v5,v6,v7,v8,col3);
	scene.addSphere(2,10,5,2.5,col1);
	scene.addSphere(6,9,3,3,col4);
	scene.addSphere(-5,6,0,2,col2);
	scene.addSphere(-9,8,3,3,col3);

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
	Ray ray(lightRay, scene);
	return ray.raytrace();
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
				SDL_SetRenderDrawColor(renderer, (int)col.r, (int)col.g, (int)col.b, 255);
			}else{
				SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
			}
			
			SDL_RenderFillRect(renderer, &r);
		}
	}
}