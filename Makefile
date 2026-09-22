# CUDA at Scale Independent Project — Batch RGB → Grayscale
#
# Build:  make
# Run:    make run           (uses data/input → data/output)
# Clean:  make clean

CXX      = nvcc
CXXFLAGS = -std=c++17 -O2 -Wno-deprecated-gpu-targets
TARGET   = batch_grayscale
SRC      = main.cu

INPUT_DIR  ?= data/input
OUTPUT_DIR ?= data/output
THREADS    ?= 16

all: build

build: $(TARGET)

$(TARGET): $(SRC) stb_image.h stb_image_write.h
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) -i $(INPUT_DIR) -o $(OUTPUT_DIR) -t $(THREADS)

clean:
	rm -f $(TARGET)
	rm -rf $(OUTPUT_DIR)

.PHONY: all build run clean
