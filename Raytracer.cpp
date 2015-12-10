#include <iostream>
#include <string>
#include "kernel.h"

using namespace std;

#define LINE_SEPARATE "----------------------------------------------------"

int main(){
	int USE_GPU, SCREEN_WIDTH, SCREEN_HEIGHT;
	string FILENAME;

	cout<<LINE_SEPARATE<<endl;
	cout<<"WELCOME TO RAYTE v0.88 CPU/GPU RAYTRACER"<<endl;
	cout<<"RAYTE IS OPEN SOURCE AND DISTRIBUTED UNDER"<<endl;
	cout<<"CC4.0-ATTRIBUTION-SHAREALIKE-INTERNATIONAL"<<endl;
	cout<<"THE SOURCE LIVES AT crynetlogistics.github.io/raytrace"<<endl;
	cout<<"RAYTE IS AN ORIGINAL PROJECT BY ZICHEN LIU"<<endl;
	cout<<LINE_SEPARATE<<endl;
	cout<<"PLEASE ENTER GPU_USAGE_LEVEL, SCREEN_WIDTH, SCREEN_HEIGHT, FILENAME:"<<endl;
	cin>>USE_GPU>>SCREEN_WIDTH>>SCREEN_HEIGHT>>FILENAME;

	raytrace(USE_GPU, SCREEN_WIDTH, SCREEN_HEIGHT, FILENAME);
}