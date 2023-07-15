# Makefile for pybind11
ifeq ($(OS),Windows_NT)
	current_os := Windows
else
	current_os := $(shell uname -s)
endif

ifeq ($(current_os),Darwin)
DARWIN_LINKER_FLAGS = -undefined dynamic_lookup
endif

CC = g++
CFLAGS = -std=c++11 -fPIC -Ofast -fopenmp
LINKER_FLAGS = -shared $(DARWIN_LINKER_FLAGS)
INCLUDE = $(shell python3 -m pybind11 --includes)
SUFFIX_PYBIND = $(shell python3-config --extension-suffix)
# Get all .cpp files from the directory above
CPPS = $(wildcard ../*.cpp)
# Exclude main.cpp from the directory above
CPPS := $(filter-out ../main.cpp, $(CPPS))

Solver$(SUFFIX_PYBIND): pybind_wrapper.cpp
	$(CC) $(CFLAGS) $(LINKER_FLAGS) $(INCLUDE) $(CPPS) $^ -o ../../python/Solver$(SUFFIX_PYBIND)

clean:
	rm Solver$(SUFFIX_PYBIND)
