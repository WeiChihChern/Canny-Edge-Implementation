CC = g++
CFLAGS = -std=c++17 -march=native -Wall -pthread -O3 -g -mavx -mavx2 -fopt-info-optimized -ffast-math -fno-math-errno

DEBUG ?= 0
ifeq ($(DEBUG), 0)
CFLAGS += -fopenmp
endif



SRCS = $(wildcard ./*.cpp)
PROG = cannyExec


INCLUDE = -I/usr/local/include -I/usr/local/include/opencv4
LIBS    = -L/usr/local/lib -lopencv_world 

OBJECTS = $(patsubst %.cpp,%.o,$(SRCS))

all: $(SRCS) $(PROG)

%.o : %.cpp
	@echo Compiling: $<
	@$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDE) $(LIBS)

$(PROG) : $(OBJECTS)
	@echo Linking: $@
	@$(CC) -std=c++11 -fopenmp -o $@ $(OBJECTS) $(INCLUDE) $(LIBS)

clean:
	rm -f $(PROG)
	rm -f $(OBJECTS)