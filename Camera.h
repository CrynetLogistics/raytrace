#pragma once
#include "vector_t.h"
class Camera
{
private:
	vector_t central_direction;
	float gridSize;
public:
	float getGridSize(void);
	vector_t getLocDir(void);
	Camera(void);
	~Camera(void);
};

