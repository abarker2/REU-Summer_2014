//--------------------------------------------------------------------------------------------
// Author: Adam Barker			email: abarker2@uccs.edu
//
// File: noop_jacobi_5pt.cu		date: June 23, 2014
//
// This program performs a simple averaging of node values using a 5-point 2D Jacobi stencil.
// This program also incorporates the use of shared memory to speed up memory accesses and
// staggers reads of the halo regions so that race conditions do not exist among threads.
//
// This program contains no advanced optimizations.
//--------------------------------------------------------------------------------------------
#include <stdio.h>
#include <output.c>

#define K 16    // Thread block dimension (K * K)
#define MIDDLE N/2 + N*(N/2)

__global__ void kernel(float * d_data, const int N)
{
	// get global x and y values
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	
	// Get local x and y values in the block and offset by halo radius.
	int tx = threadIdx.x + 1;
	int ty = threadIdx.y + 1;
	
	// Place local nodes into local variables (registers)	
	float curr;	 // current node
	float right;     // node right of the current node
	float left;	 // node left of the current node
	float up;	 // node above the current node
	float down;	 // node below the current node

	// value to multiply node values by when performing stencil computation (average).
	float coef = 0.2f;

	// Allocate shared memory to include halo region.
	__shared__ float s_data[K+2][K+2];

	// get current node from global memory and place into local variable.
	curr = d_data[ix + iy*N];

	// place current node into shared memory so other threads can access it.
	s_data[ty][tx] = curr;
	__syncthreads();

	// Place left halo into shared memory
	if(tx == 1) {
		s_data[ty][tx-1] = d_data[(ix-1) + iy*N];
	}
	__syncthreads();

	// Place lower halo into shared memory
	if(ty == 1) {
		s_data[ty-1][tx] = d_data[ix + (iy-1)*N];
	}
	__syncthreads();

	// Place right halo into shared memory
	if(tx == K) {
		s_data[ty][tx+1] = d_data[(ix+1) + iy*N];
	}
	__syncthreads();

	// Place upper halo into shared memory
	if(ty == K) {
		s_data[ty+1][tx] = d_data[ix + (iy+1)*N];
	}
	__syncthreads();

	// Retrieve local nodes from shared memory and place into local variables.
	right = s_data[ty][tx+1];
	left  = s_data[ty][tx-1];
	up    = s_data[ty+1][tx];
	down  = s_data[ty-1][tx];
	__syncthreads();

	// Don't try to perform calculations on edges of data
	if(ix == 0 || iy == 0 || ix == N-1 || iy == N-1) return;
	
	// Keep input static so that values go up over time. (e.g. constant heat)
	if(ix + iy*N == MIDDLE) return;
	
	// Compute output value
	curr = coef * (curr + right + left + up + down);
	__syncthreads();
	
	// update shared and global memory with new values.
	s_data[ty][tx] = curr;
	__syncthreads();

	d_data[ix + iy*N] = s_data[ty][tx];
	__syncthreads();
}

int main(int argc, char* *argv)
{
	// Make sure enough inputs were given by the user.
	if(argc != 3) { printf("USAGE: %s <block> <steps>\n", argv[0]); return 10; }

	const int N     = atoi(argv[1]);
	const int STEPS = atoi(argv[2]);
	
	// Set the number of blocks in the grid and the number of threads per block.
	const dim3 blocks (N/K, N/K);
	const dim3 threads(K, K);

	const int ARRAY_BYTES  = N * N * sizeof(float);
	
	// These variables are the host array and the device array for doing computations on.
	float * h_data;
	float * d_data;

	// Allocate data for the arrays
	h_data = (float*)malloc(ARRAY_BYTES);
	cudaMalloc(&d_data, ARRAY_BYTES);

	// Place initial values into h_data
	for(int j=0; j<N; j++) {
		for(int i=0; i<N; i++) {
			h_data[i + j*N] = 5.0f;
		}
	}
	
	// place an initial higher value into the middle of the array.
	h_data[MIDDLE] = 10.0f;
	
	// Copy h_data onto the device's array d_data
	cudaMemcpy(d_data, h_data, ARRAY_BYTES, cudaMemcpyHostToDevice);
	
	// Start kernel and iterate for the given number of steps.
	for(int i=0; i<STEPS; i++) kernel<<<blocks, threads>>>(d_data, N);

	// Copy d_data back into the host's array h_data
	cudaMemcpy(h_data, d_data, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	// Output h_data
	output("out.image", h_data, N, N, 1);
	
	free(h_data);
	cudaFree(d_data);

	return 0;
}
