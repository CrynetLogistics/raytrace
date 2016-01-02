#pragma once

//A queue implemented using the
//circular array strategy
template <class T>
class Stack
{
private:
	int maxHeight;
	
	//points to the first free element
	int top;

	T* stackArray;
public:
	__host__ __device__ Stack(int maxHeight);
	__host__ __device__ void add(T item);
	__host__ __device__ T pop(void);
	__host__ __device__ bool isEmpty(void);
	__host__ __device__ ~Stack(void);
};

template <class T>
__host__ __device__ Stack<T>::Stack(int maxHeight){
	this->maxHeight = maxHeight;
	stackArray = (T*)malloc(sizeof(T)*maxHeight);
	top = 0;
}

template <class T>
__host__ __device__ void Stack<T>::add(T item){
	if(top==maxHeight){
		printf("Error: Stack Full, ");
		return;
	}
	stackArray[top] = item;
	top++;
}

template <class T>
__host__ __device__ T Stack<T>::pop(void){
	if(top==0){
		printf("Error: Stack Empty, ");
	}

	top--;

	return stackArray[top];
}

template <class T>
__host__ __device__ bool Stack<T>::isEmpty(void){
	return top==0;
}


template <class T>
__host__ __device__ Stack<T>::~Stack(void){
	free(stackArray);
}
