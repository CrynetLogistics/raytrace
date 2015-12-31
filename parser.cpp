#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include "parser.h"
#include "Auxiliary/structures.h"
using namespace std;

scenePrototype_t parseFile(string filename) {
	ifstream inputFile;
	scenePrototype_t output;
	inputFile.open(filename);
	if(inputFile.is_open()){
		//string s;
		char temC;
		char* s = (char*)malloc(sizeof(char)*512);
		//char head; //identifier for type

		int numberOfVerticies = 0;
		vertex_t* verts = (vertex_t*)malloc(sizeof(vertex_t)*numberOfVerticies);
		int numberOfTris = 0;
		triPrototype_t* tris = (triPrototype_t*)malloc(sizeof(triPrototype_t)*numberOfTris);

		while(!inputFile.eof()){
			char i = inputFile.peek();
			if(i=='#'||i=='s'||i=='o'){
				inputFile.getline(s, 512);
			}else if(i=='v'){
				numberOfVerticies++;
				verts = (vertex_t*)realloc(verts, sizeof(vertex_t)*numberOfVerticies);
				inputFile>>temC;
				inputFile>>verts[numberOfVerticies-1].x>>verts[numberOfVerticies-1].y>>verts[numberOfVerticies-1].z;
				inputFile>>ws;
			}else if(i=='f'){
				numberOfTris++;
				tris = (triPrototype_t*)realloc(tris, sizeof(triPrototype_t)*numberOfTris);
				inputFile>>temC;
				inputFile>>tris[numberOfTris-1].v1>>tris[numberOfTris-1].v2>>tris[numberOfTris-1].v3;
				inputFile>>ws;
			}else{
				cerr<<"I DONT KNOW WHAT I HAVE ENCOUNTERED"<<endl;
				inputFile.getline(s, 512);
			}
		}

		inputFile.close();

		for(int i=0; i<numberOfVerticies; i++){
			cout<<"Vertex "<<i<<" : ("<<verts[i].x<<","<<verts[i].y<<","<<verts[i].z<<")"<<endl;
		}
		for(int i=0; i<numberOfTris; i++){
			cout<<"Tri "<<i<<" : ("<<tris[i].v1<<","<<tris[i].v2<<","<<tris[i].v3<<")"<<endl;
		}

		output.tris = tris;
		output.verts = verts;
		output.numOfTris = numberOfTris;
		output.numOfVerts = numberOfVerticies;
		free(s);
	}else{
		cerr<<"ERROR: FILE CANNOT BE OPENED"<<endl;
		int dummy;
		cin>>dummy;
		exit(0);
	}
	//VERTS SHOULD BE FREED AT THE END OF THE PROGRAM free(verts);

	return output;
}
