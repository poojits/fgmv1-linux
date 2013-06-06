all: train predict

train: FGM.o MKL.o linear.o
	g++ -g train.cpp FGM.o MKL.o linear.o -o train -lm

predict: FGM.o MKL.o linear.o
	g++ -g predict.cpp FGM.o MKL.o linear.o -o predict -lm
	
FGM.o: FGM.cpp FGM.h
	g++ -c -g FGM.cpp -o FGM.o

MKL.o: MKL.cpp MKL.h
	g++ -c -g MKL.cpp -o MKL.o

linear.o: linear.cpp linear.h
	g++ -c -g linear.cpp -o linear.o

clean:
	rm -rf *.o train predict
