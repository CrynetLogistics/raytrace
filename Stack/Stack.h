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
	Stack(int maxHeight);
	void add(T item);
	T pop(void);
	bool isEmpty(void);
	~Stack(void);
};

template <class T>
Stack<T>::Stack(int maxHeight){
	this->maxHeight = maxHeight;
	stackArray = (T*)malloc(sizeof(T)*maxHeight);
	top = 0;
}

template <class T>
void Stack<T>::add(T item){
	if(top==maxHeight){
		printf("Error: Stack Full, ");
		return;
	}
	stackArray[top] = item;
	top++;
}

template <class T>
T Stack<T>::pop(void){
	if(top==0){
		printf("Error: Stack Empty, ");
	}

	top--;

	return stackArray[top];
}

template <class T>
bool Stack<T>::isEmpty(void){
	return top==0;
}


template <class T>
Stack<T>::~Stack(void){
	free(stackArray);
}
