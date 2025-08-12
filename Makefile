NVCC = nvcc
CXX = g++
CFLAGS = -std=c++11

TARGET = cuda_program

HOST_SRC = host.cpp
KERNEL_SRC = kernels.cu

all: $(TARGET)

$(TARGET): $(HOST_SRC) $(KERNEL_SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(HOST_SRC) $(KERNEL_SRC)

clean:
	rm -f $(TARGET)