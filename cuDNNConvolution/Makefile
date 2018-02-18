CXX := nvcc
CXXFLAGS := -arch=sm_35 -std=c++11 -O2
HEADERS := -I/usr/local/cuda-8.0/include `pkg-config --cflags opencv`
LIBS := -L/usr/local/cuda-8.0/lib64 `pkg-config --libs opencv` \
		`pkg-config --libs openblas` -lcudnn -lopenblas

SRC := main.cu

TARGET := conv

all: conv

conv: $(SRC)
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(SRC) -o $(TARGET)

.phony: clean

clean:
	rm -f $(TARGET)
