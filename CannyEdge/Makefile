CC = g++ 
CFLAGS = -std=c++11 -march=native -Wall -pthread -O2 -g -mavx2 -fopt-info-vec-optimized -ffast-math -fno-math-errno

DEBUG ?= 0
ifeq ($(DEBUG), 0)
CFLAGS += -fopenmp
endif



SRCS = $(wildcard ./*.cpp)
PROG = cannyExec

OPENCV = -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_world
LIBS = $(OPENCV)

OBJECTS = $(patsubst %.cpp,%.o,$(SRCS))

all: $(SRCS) $(PROG)

%.o : %.cpp
	@echo Compiling: $<
	@$(CC) $(CFLAGS) -c $< -o $@ $(OPENCV)

$(PROG) : $(OBJECTS)
	@echo Linking: $@
	@$(CC) -std=c++11 -fopenmp -o $@ $(OBJECTS) $(OPENCV)

clean:
	rm -f $(PROG)
	rm -f $(OBJECTS)