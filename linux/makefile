CC=nvcc
CC2=g++
FLAGS =  -c -rdc=true -std=c++11

all: raytracer

raytracer: Camera.o kernel.o Light.o Material.o Mesh.o parser.o Plane.o Ray.o Raytracer.o Scene.o Sphere.o Tri.o extremum_t.o vector_t.o BinTree.o BinTreeNode.o
	$(CC) Camera.o kernel.o Light.o Material.o Mesh.o parser.o Plane.o Ray.o Raytracer.o Scene.o Sphere.o Tri.o extremum_t.o vector_t.o BinTree.o BinTreeNode.o -o raytracer -rdc=true -I/sdl_src/include -L./sdl_src/lib -lSDL2 -lSDL2_image

Camera.o: ../Camera.cu
	$(CC) $(FLAGS) ../Camera.cu

kernel.o: ../kernel.cu
	$(CC) $(FLAGS) ../kernel.cu

Light.o: ../Light.cu
	$(CC) $(FLAGS) ../Light.cu

Material.o: ../Material.cu
	$(CC) $(FLAGS) ../Material.cu

Mesh.o: ../Mesh.cu
	$(CC) $(FLAGS) ../Mesh.cu

parser.o: ../parser.cu
	$(CC) $(FLAGS) ../parser.cu

Plane.o: ../Plane.cu
	$(CC) $(FLAGS) ../Plane.cu

Ray.o: ../Ray.cu
	$(CC) $(FLAGS) ../Ray.cu

Raytracer.o: ../Raytracer.cu
	$(CC) $(FLAGS) ../Raytracer.cu

Scene.o: ../Scene.cu
	$(CC) $(FLAGS) ../Scene.cu

Sphere.o: ../Sphere.cu
	$(CC) $(FLAGS) ../Sphere.cu

Tri.o: ../Tri.cu
	$(CC) $(FLAGS) ../Tri.cu

extremum_t.o: ../Auxiliary/extremum_t.cu
	$(CC) $(FLAGS) ../Auxiliary/extremum_t.cu

vector_t.o: ../Auxiliary/vector_t.cu
	$(CC) $(FLAGS) ../Auxiliary/vector_t.cu

BinTree.o: ../BinTree/BinTree.cu
	$(CC) $(FLAGS) ../BinTree/BinTree.cu

BinTreeNode.o: ../BinTree/BinTreeNode.cu
	$(CC) $(FLAGS) ../BinTree/BinTreeNode.cu

clean:
	rm *.o
