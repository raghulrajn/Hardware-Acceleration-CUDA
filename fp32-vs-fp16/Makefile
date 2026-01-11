NVCC = nvcc
CXXFLAGS = -O3
TARGET = fp32

all: $(TARGET)

$(TARGET): main.cpp
	$(NVCC) $(CXXFLAGS) main.cpp -lcublas -o $(TARGET)

clean:
	rm -f $(TARGET)
