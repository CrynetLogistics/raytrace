#pragma once

//A queue implemented using the
//circular array strategy
template <class T>
class Queue
{
private:
	int maxLength;
	int headPosition;
	int firstEmptyPosition;
	int size;
	T* queueArray;
public:
	Queue(int maxLength);
	void add(T item);
	T pop(void);
	bool isEmpty(void);
	~Queue(void);
};

template <class T>
Queue<T>::Queue(int maxLength){
	this->maxLength = maxLength;
	queueArray = (T*)malloc(sizeof(T)*maxLength);
	headPosition = 0;
	firstEmptyPosition = 0;
	size = 0;
}

template <class T>
void Queue<T>::add(T item){
	if(size==maxLength){
		printf("Error: Queue Full, ");
	}
	queueArray[firstEmptyPosition] = item;
	
	firstEmptyPosition++;
	if(firstEmptyPosition>=maxLength){
		firstEmptyPosition = 0;
	}

	size++;
}

template <class T>
T Queue<T>::pop(void){
	if(size==0){
		printf("Error: Queue Empty, ");
	}
	headPosition++;
	if(headPosition>=maxLength){
		headPosition = 0;
	}

	size--;
	if(headPosition==0){
		return queueArray[maxLength-1];
	}else{
		return queueArray[headPosition-1];
	}
}

template <class T>
bool Queue<T>::isEmpty(void){
	return size==0;
}

template <class T>
Queue<T>::~Queue(void){
	free(queueArray);
}
