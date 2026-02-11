CXX = g++
# -mavx2 -mfma -march=native: Hardware-specific vectorization
CXXFLAGS = -std=c++17 -mavx2 -mfma -march=native -Wall -Wextra

TARGET = bench
SRC = cpu_performance.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run