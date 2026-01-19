NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_86
TARGET = fp32

all: $(TARGET)

$(TARGET): main.o kernels.o
	$(NVCC) main.o kernels.o -lcublas -o $(TARGET)
	rm -f main.o kernels.o

main.o: main.cpp kernels.h
	$(NVCC) $(NVCCFLAGS) -c main.cpp -o main.o

kernels.o: kernels.cu kernels.h
	$(NVCC) $(NVCCFLAGS) -c kernels.cu -o kernels.o

clean:
	rm -f *.o $(TARGET)
