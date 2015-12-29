#include "extremum_t.cuh"

__host__ __device__ extremum_t::extremum_t(extremum_t* ex){//copy constructor
	vertex_t low = ex->getLowExtremum;
	vertex_t high = ex->getHighExtremum;

	lowX = low.x;
	lowY = low.y;
	lowZ = low.z;
	highX = high.x;
	highY = high.y;
	highZ = high.z;
}

__host__ __device__ extremum_t::extremum_t(vertex_t e){
	lowX = e.x;
	highX = e.x;
	lowY = e.y;
	highY = e.y;
	lowZ = e.z;
	highZ = e.z;
}

__host__ __device__ extremum_t::extremum_t(vertex_t e1, vertex_t e2)
{
	if(e1.x > e2.x){
		lowX = e2.x;
		highX = e1.x;
	}else{
		lowX = e1.x;
		highX = e2.x;
	}

	if(e1.y > e2.y){
		lowY = e2.y;
		highY = e1.y;
	}else{
		lowY = e1.y;
		highY = e2.y;
	}

	if(e1.z > e2.z){
		lowZ = e2.z;
		highZ = e1.z;
	}else{
		lowZ = e1.z;
		highZ = e2.z;
	}
}

__host__ __device__ void extremum_t::mergeExtrema(extremum_t ex){
	vertex_t ex_low = ex.getLowExtremum();
	vertex_t ex_high = ex.getHighExtremum();
	
	if(ex_low.x<lowX){
		lowX = ex_low.x;
	}

	if(ex_low.y<lowY){
		lowY = ex_low.y;
	}

	if(ex_low.z<lowZ){
		lowZ = ex_low.z;
	}

	if(ex_high.x>highX){
		highX = ex_high.x;
	}

	if(ex_high.y>highY){
		highY = ex_high.y;
	}

	if(ex_high.z>highZ){
		highZ = ex_high.z;
	}
}


__host__ __device__ void extremum_t::factorInExtremum(vertex_t e){
	if(e.x>highX){
		highX = e.x;
	}

	if(e.x<lowX){
		lowX = e.x;
	}

	if(e.y>highY){
		highY = e.y;
	}

	if(e.y<lowY){
		lowY = e.y;
	}

	if(e.z>highZ){
		highZ = e.z;
	}

	if(e.z<lowZ){
		lowZ = e.z;
	}
}

__host__ __device__ vertex_t extremum_t::getLowExtremum(void){
	vertex_t v;
	v.x = lowX;
	v.y = lowY;
	v.z = lowZ;
	return v;
}

__host__ __device__ vertex_t extremum_t::getHighExtremum(void){
	vertex_t v;
	v.x = highX;
	v.y = highY;
	v.z = highZ;
	return v;
}

__host__ __device__ extremum_t::~extremum_t(void)
{
}
