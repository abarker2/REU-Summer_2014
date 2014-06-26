//--------------------------------------------------------------------------------------------
// Author: Adam Barker			email: abarker2@uccs.edu
//
// File: noop_jacobi_7pt.cu		date: June 24, 2014
//
// This program performs a simple averaging of node values using a 7-point 3D Jacobi stencil.
// This program also incorporates the use of shared memory to speed up memory accesses and
// staggers reads of the halo regions so that race conditions do not exist among threads.
//
// This program contains no advanced optimizations.
//--------------------------------------------------------------------------------------------
#include <stdio.h>
#include <output.c>

#define K 16
#define Z 2 
#define MIDDLE N/2 + N*(N/2) + N*N*(N/2)

__global__ void kernel(float * d_data, const int N)
{
	// global thread coordinates for reads and writes
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int iz = threadIdx.z + blockDim.z * blockIdx.z;
	
	// local thread coordinates within block
	// offset by 1 to encompass halo region
	int tx = threadIdx.x + 1;
	int ty = threadIdx.y + 1;
	int tz = threadIdx.z + 1;

	// local variables for node dependencies
	float curr;     // Current node
	float right;    // node right of this node
	float left;     // node left of this node
	float up;       // node above this node
	float down;     // node below this node
	float front;    // node in front of this node
	float back;     // node behind this node
	
	// number to multiply by for calculation (average)
	float coef = (1.0f/7.0f);

	// Shared memory allocation to make reads faster.
	// shared memory is accessed in row-major order (z, y, x) to
	// keep memory accesses coalesced for faster reads.
	__shared__ float s_data[Z+2][K+2][K+2];

	// fetch current node value from global memory
	// and place into local and shared memory
	curr = d_data[ix + iy*N + iz*N*N];
	s_data[tz][ty][tx] = curr;
	__syncthreads();

	// Place right and left halos into smem.
	if(tx == 1) {
        if(ix > 0)
		s_data[tz][ty][tx-1] = d_data[(ix-1) + iy*N + iz*N*N];
	}
	if(tx == K) {
        if(ix < N-1)
		s_data[tz][ty][tx+1] = d_data[(ix+1) + iy*N + iz*N*N];
	}
	__syncthreads();

	// Place the halos that are above and below into smem.
	if(ty == 1) {
        if(iy > 0)
		s_data[tz][ty-1][tx] = d_data[ix + (iy-1)*N + iz*N*N];
	}
	if(ty == K) {
        if(iy < N-1)
		s_data[tz][ty+1][tx] = d_data[ix + (iy+1)*N + iz*N*N];
	}
	__syncthreads();
	
	//Place halos that are in front and behind into smem.
	if(tz == 1) {
        if(iz > 0)
		s_data[tz-1][ty][tx] = d_data[ix + iy*N + (iz-1)*N*N];
	}
	if(tz == Z) {
        if(iz < N-1)
		s_data[tz+1][ty][tx] = d_data[ix + iy*N + (iz+1)*N*N];
	}
	__syncthreads();

	// Read in data from smem into local variables.
	right = s_data[tz][ty][tx+1];
	left  = s_data[tz][ty][tx-1];
	up    = s_data[tz][ty+1][tx];
	down  = s_data[tz][ty-1][tx];
	front = s_data[tz+1][ty][tx];
	back  = s_data[tz-1][ty][tx];
	__syncthreads();

	// Don't compute output for nodes that are on the edge or in the middle of the data.
	if( (ix == 0 || ix == N-1) || (iy == 0 || iy == N-1) || (iz == 0 || iz == N-1) ||
		(ix + iy*N + iz*N*N == MIDDLE) )
		return;

	// Compute output value and write to smem.
	curr = coef * (curr + right + left + up + down + front + back);
	__syncthreads();

	s_data[tz][ty][tx] = curr;
	__syncthreads();
	
	// Write value back to global memory.
	d_data[ix + iy*N + iz*N*N] = s_data[tz][ty][tx];
	__syncthreads();
}

int main(int argc, char* *argv)
{
    if(argc != 3) {printf("USAGE: %s <blocks> <steps>\n", argv[0]); return 10;} 

    const int N     = atoi(argv[1]);    // Dimension of block (N * N)
    const int STEPS = atoi(argv[2]);    // Number of iterations to perform.

    // constants to hold grid and threadblock dimensions
    const dim3 blocks ( N/K, N/K, N/Z );
    const dim3 threads( K, K, Z );
    
    // constant to hold size of data in bytes
    const int ARRAY_BYTES = N * N * N * sizeof(float);

    // arrays to hold the data to perform compuation on.
    float * h_data;
    float * d_data;

    h_data = (float*)malloc(ARRAY_BYTES);
    cudaMalloc(&d_data, ARRAY_BYTES);

    // Initialize array
    for(int k=0; k < N; k++) {
        for(int j=0; j<N; j++) {
            for(int i=0; i<N; i++) {
                h_data[i + j*N + k*N*N] = 5.0f;
            }
        }
    }
    
    // Place a differing value into middle of array that will spread.
    h_data[MIDDLE] = 10.0f;

    // Copy data to the device.
    cudaMemcpy(d_data, h_data, ARRAY_BYTES, cudaMemcpyHostToDevice);

    for(int i=0; i<STEPS; i++) kernel<<<blocks, threads>>>(d_data, N);
   
    // Copy data back from device to the host.
    cudaMemcpy(h_data, d_data, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Output the data into out.image
    output("out.image", h_data, N, N, N);

    free(h_data);
    cudaFree(d_data);

    return 0;
}	
