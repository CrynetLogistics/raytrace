#include <iostream>
#include <string>
#include "kernel.h"
#include "Auxiliary\structures.h"

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


	cudaGetDeviceCount(&CUDA_DEVICE_NUMBER);

	cout<<LINE_SEPARATE<<endl;
	cout<<"WELCOME TO RAYTE v0.96 CPU/GPU RAYTRACER"<<endl;
	cout<<"RAYTE IS OPEN SOURCE AND DISTRIBUTED UNDER"<<endl;
	cout<<"CC4.0-ATTRIBUTION-SHAREALIKE-INTERNATIONAL"<<endl;
	cout<<"THE SOURCE LIVES AT crynetlogistics.github.io/raytrace"<<endl;
	cout<<"RAYTE IS AN ORIGINAL PROJECT BY ZICHEN LIU"<<endl;
	cout<<LINE_SEPARATE<<endl;
	cout<<"SYSTEM SPECIFICATIONS AS DETECTED;"<<endl;
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
	cout<<LINE_SEPARATE<<endl;

	cout<<"PLEASE ENTER GPU_USAGE_LEVEL:";

	cin>>USE_GPU;

	cout<<"PLEASE ENTER SCREEN_WIDTH:";

	cin>>SCREEN_WIDTH;

	cout<<"PLEASE ENTER SCREEN_HEIGHT:";

	cin>>SCREEN_HEIGHT;

	cout<<"PLEASE ENTER FILENAME:";

	cin>>FILENAME;

	cout<<"PLEASE ENTER MSAA_LEVEL:";

	cin>>MSAA_LEVEL;

	raytrace(USE_GPU, SCREEN_WIDTH, SCREEN_HEIGHT, FILENAME, MSAA_LEVEL);
}