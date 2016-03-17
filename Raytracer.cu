#include <iostream>
#include <string>
#include "kernel.h"
#include "Auxiliary/structures.h"

using namespace std;

#define LINE_SEPARATE "----------------------------------------------------"

void handleCudaError(cudaError_t e){
		if(e==cudaErrorInvalidDevice){
			cout<<"UNEXPECTED DEVICE ERROR"<<endl;
		}else if(e==cudaSuccess){
		
		}else{
			cout<<"UNKNOWN ERROR"<<endl;
		}
}

int main(){
	int CUDA_DEVICE_NUMBER;
	cudaDeviceProp CUDA_DEVICE_PROP;
	int USE_GPU, SCREEN_WIDTH, SCREEN_HEIGHT, MSAA_LEVEL;
	string FILENAME;
	int BSPBVH_DEPTH, ENABLE_TEXTURES, DEBUG_LEVEL, ANIMATION_LEVEL;


	cudaGetDeviceCount(&CUDA_DEVICE_NUMBER);

	cout<<LINE_SEPARATE<<endl;
	cout<<"WELCOME TO RAYTE v1.22 CPU/GPU RAYTRACER"<<endl;
	cout<<"RAYTE IS OPEN SOURCE AND DISTRIBUTED UNDER"<<endl;
	cout<<"CC4.0-ATTRIBUTION-SHAREALIKE-INTERNATIONAL"<<endl;
	cout<<"THE SOURCE LIVES AT crynetlogistics.github.io/raytrace"<<endl;
	cout<<"RAYTE IS AN ORIGINAL PROJECT BY ZICHEN LIU"<<endl;
	cout<<LINE_SEPARATE<<endl;
	cout<<"SYSTEM SPECIFICATIONS AS DETECTED;"<<endl;
	if(CUDA_DEVICE_NUMBER>=1){
		cout<<"NUMBER OF CUDA ENABLED DEVICES DETECTED -> "<<CUDA_DEVICE_NUMBER<<endl;
		for(int i=0; i<CUDA_DEVICE_NUMBER; i++){
			cudaError_t e = cudaGetDeviceProperties(&CUDA_DEVICE_PROP, i);
			handleCudaError(e);
			cout<<"DEVICE "<<i<<" -> "<<CUDA_DEVICE_PROP.name<<endl;
			cout<<"\tCOMPUTE VERSION -> "<<CUDA_DEVICE_PROP.major<<"."<<CUDA_DEVICE_PROP.minor<<endl;
			cout<<"\tTOTAL GLOBAL MEMORY -> "<<CUDA_DEVICE_PROP.totalGlobalMem/(1024*1024)<<"MiB"<<endl;
			cout<<"\tCLOCK FREQUENCY -> "<<CUDA_DEVICE_PROP.clockRate/1024<<"MHz"<<endl;
			cout<<"\tSTREAMING MULTIPROCESSOR COUNT -> "<<CUDA_DEVICE_PROP.multiProcessorCount<<endl;
		}
	}else{
		cout<<"NO CUDA DEVICES DETECTED, USE CPU RENDERING."<<endl;
	}
	cout<<LINE_SEPARATE<<endl;

	cout<<"ENTER GPU_USAGE_LEVEL:";
	cin>>USE_GPU;

	cout<<"ENTER SCREEN_WIDTH:";
	cin>>SCREEN_WIDTH;

	cout<<"ENTER SCREEN_HEIGHT:";
	cin>>SCREEN_HEIGHT;

	cout<<"ENTER FILENAME:";
	cin>>FILENAME;

	cout<<"ENTER MSAA_LEVEL:";
	cin>>MSAA_LEVEL;

	cout<<"ENTER BSPBVH_DEPTH:";
	cin>>BSPBVH_DEPTH;

	cout<<"ENTER TEXTURE SYSTEM INITIALISATION LEVEL:";
	cin>>ENABLE_TEXTURES;

	cout<<"ENTER DEBUG LEVEL:";
	cin>>DEBUG_LEVEL;

	cout<<"ENTER RENDER MODE, IMAGE(0), ANIMATION(1):";
	cin>>ANIMATION_LEVEL;

	if(ANIMATION_LEVEL==0){
		raytrace(USE_GPU, SCREEN_WIDTH, SCREEN_HEIGHT, FILENAME, MSAA_LEVEL, BSPBVH_DEPTH, ENABLE_TEXTURES, DEBUG_LEVEL);
	}else{

		int FRAMES_TO_RENDER, START_FRAME;

		cout<<"ENTER THE STARTING INDEX:";
		cin>>START_FRAME;

		cout<<"ENTER THE NUMBER OF FRAMES TO RENDER:";
		cin>>FRAMES_TO_RENDER;


		for(int i=START_FRAME; i<START_FRAME+FRAMES_TO_RENDER; i++){
			string frameName = "";
			frameName.append(FILENAME);
            if(i<=9){
				frameName.append("_00000");
            }else if(i<=99){
				frameName.append("_0000");
			}else if(i<=999){
				frameName.append("_000");
			}else{
				frameName.append("_00");
            }
			frameName.append(to_string(i));
			frameName.append(".obj");
			raytrace(USE_GPU, SCREEN_WIDTH, SCREEN_HEIGHT, frameName, MSAA_LEVEL, BSPBVH_DEPTH, ENABLE_TEXTURES, DEBUG_LEVEL);
		}
	}
}
