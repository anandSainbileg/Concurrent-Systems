# width height kernel_order nchannels nkernels 
ARGS=100 100 5 128 128

build:
	gcc -O3 -msse4 -fopenmp conv-harness.c
run:
	./a.out $(ARGS)
all:
	make build && make run
all-big:
	make build && make run ARGS="128 128 7 256 256"
