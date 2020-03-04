CC = g++
# CFLAGS = -std=c++17 -march=native -Wall -pthread -O2 -g -pg -mavx2 -ffast-math -fno-math-errno #-fopt-info-optimized 
CFLAGS = -std=c++17 -march=native -Wall -pthread -O2 -g -pg -mavx2 -fno-omit-frame-pointer -ffast-math -DNDEBUG -fno-default-inline -fno-inline

DEBUG ?= 0
ifeq ($(DEBUG), 0)
# CFLAGS += -fopenmp
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
	@$(CC) -std=c++11 -g -pg -fopenmp -o $@ $(OBJECTS) $(INCLUDE) $(LIBS)

clean:
	rm -f $(PROG)
	rm -f $(OBJECTS)