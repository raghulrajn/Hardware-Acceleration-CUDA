NVCC = nvcc
CXXFLAGS = -O3
TARGET = cuda_check

all: $(TARGET)

$(TARGET): main.o kernels.o
	$(NVCC) main.o kernels.o -o $(TARGET)
	rm -f main.o kernels.o

main.o: main.cpp kernels.h
	$(NVCC) -c main.cpp -o main.o

kernels.o: kernels.cu kernels.h
	$(NVCC) -c kernels.cu -o kernels.o

clean:
	rm -f *.o $(TARGET)