#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I../../include

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

FLAGS= -Xcompiler "-fopenmp -g" -std=c++11 -g

ARCH= -arch sm_52

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):../../libWin32:../../libx64
    LIBS= -lcuda -L ../../libWin32 -L ../../libx64 -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):../../lib
        LIBS= -Xlinker -framework -Xlinker cuda -L ../../lib -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):../../lib:../../lib64
        LIBS= -lcuda -L ../../lib -L ../../lib64 -lcupti
    endif
    OBJ = o
endif

CUBIN = cubin

all: vec external_id.$(OBJ) vec.$(OBJ) vecSub.$(CUBIN) vecAdd.$(CUBIN)

vec: external_id.$(OBJ) vec.$(OBJ)
	nvcc -o $@ $^ $(FLAGS) $(ARCH) $(LIBS)

external_id.$(OBJ): external_id.cpp
	nvcc -c $(INCLUDES) $(FLAGS) $(ARCH) $<

vec.$(OBJ): vec.cu
	nvcc -c $(INCLUDES) $(FLAGS) $(ARCH) $<

vecSub.$(CUBIN): vecSub.cu
	nvcc -cubin $(INCLUDES) $(FLAGS) $(ARCH) $<

vecAdd.$(CUBIN): vecAdd.cu
	nvcc -cubin $(INCLUDES) $(FLAGS) $(ARCH) $<

run: vec
	./$<

clean:
	rm -f vec external_id.$(OBJ) vec.$(OBJ) vecSub.$(CUBIN) vecAdd.$(CUBIN)
