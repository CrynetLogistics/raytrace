//============================================================================
// Name        : parser.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "Auxiliary/structures.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
using namespace std;

int main() {

	ifstream inputFile;
	inputFile.open("world1.rayte");
	if(inputFile.is_open()){
		string s;

		inputFile>>s;//Verticies:
		int numberOfVerticies;
		inputFile>>numberOfVerticies;
		vertex_t* verts = (vertex_t*)malloc(sizeof(vertex_t)*numberOfVerticies);
		for(int i=0; i<numberOfVerticies; i++){
			inputFile>>verts[i].x>>verts[i].y>>verts[i].z;
		}

		inputFile>>s;//Colours:
		int numberOfColours;
		inputFile>>numberOfColours;
		colour_t* cols = (colour_t*)malloc(sizeof(colour_t)*numberOfColours);
		for(int i=0; i<numberOfColours; i++){
			inputFile>>cols[i].r>>cols[i].g>>cols[i].b;
			if(cols[i].r>255||cols[i].r<0||cols[i].g>255||cols[i].g<0||cols[i].b>255||cols[i].b<0){
				cout<<"WARNING: COLOUR PARSED IS NOT IN THE RANGE OF [0,255]"<<endl;
			}
		}



		inputFile.close();


		for(int i=0; i<numberOfVerticies; i++){
			cout<<"Vertex "<<i<<" : ("<<verts[i].x<<","<<verts[i].y<<","<<verts[i].z<<")"<<endl;
		}
		for(int i=0; i<numberOfColours; i++){
			cout<<"Colour "<<i<<" : ("<<cols[i].r<<","<<cols[i].g<<","<<cols[i].b<<")"<<endl;
		}


	}else{
		cout<<"ERROR: FILE CANNOT BE OPENED"<<endl;
	}
	//VERTS SHOULD BE FREED AT THE END OF THE PROGRAM free(verts);
	return 0;
}
