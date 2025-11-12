# Simple Makefile for final-project
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall
SRC := tile-selection.cpp
TARGET := tile-selection

.PHONY: all clean run visualize
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

visualize: run
	python3 visualize.py

clean:
	rm -f $(TARGET) *.o