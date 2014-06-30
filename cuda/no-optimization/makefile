INCLUDE="../src/include"

all:
	nvcc noop_jacobi_5pt.cu -o $(HOME)/bin/5pt -I$(INCLUDE)
	nvcc noop_jacobi_7pt.cu -o $(HOME)/bin/7pt -I$(INCLUDE)
	nvcc noop_jacobi_27pt.cu -o $(HOME)/bin/27pt -I$(INCLUDE)
