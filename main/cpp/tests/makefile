# Compiler
CC = g++

# Compiler Flags
CXXFLAGS = -std=c++14 -Wall -I../

# Linker Flags
# Adjust the path of libgtest.a and libgtest_main.a if necessary
LDFLAGS = -L/usr/local/lib -lgtest -lgtest_main -pthread

# Source files
SRCS = $(wildcard *.cpp ../VicsekModel.cpp ../SwarmModel.cpp ../Particle.cpp ../helperFunctions.cpp)
OBJS = $(SRCS:.cpp=.o)
EXEC = run_tests

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $^ $(LDFLAGS) -o $@

%.o: %.cpp
	$(CC) -c $(CXXFLAGS) $< -o $@

run: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(OBJS) $(EXEC)
