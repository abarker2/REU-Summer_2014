//--------------------------------------------------------------------------------------------
// Author: Adam Barker			email: abarker2@uccs.edu
//
// File: noop_jacobi_27pt.cu		date: June 24, 2014
//
// This program performs a simple averaging of node values using a 27-point 3D Jacobi stencil.
// This program also incorporates the use of shared memory to speed up memory accesses and
// staggers reads of the halo regions so that race conditions do not exist among threads.
//
// This program contains no advanced optimizations.
//--------------------------------------------------------------------------------------------
#include <stdio.h>
#include <output.c>

#define CURRENT ix + iy*N + iz*N*N
#define MIDDLE  N/2 + N*(N/2) + N*N*(N/2)

#define Z 4
#define Y 8
#define X 8

__global__ void kernel(float * d_data, const int N)
{
    // Global data coordinate variables
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    // Local data coordinate variables
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    // Shared memory allocation
    __shared__ float s_data[Z+2][Y+2][X+2];

    // Current node
    float curr;

    // Set 1: nodes 1 unit away [6]
    float right;
    float left;
    float up;
    float down;
    float front;
    float back;

    // Set 2: nodes sqrt(2) units away (midpoints) [12]
    float right_front;      float right_up;         float front_up;
    float right_back;       float right_down;       float front_down;
    float left_front;       float left_up;          float back_up;
    float left_back;        float left_down;        float back_down;

    // Set 3: nodes sqrt(3) units away (corners) [8]
    float right_front_up;   float left_front_up;
    float right_front_down; float left_front_down;
    float right_back_up;    float left_back_up;
    float right_back_down;  float left_back_down;

    // Array to hold coefficients to multiply certain node groups by [close, med, far]
    float coef = (1.0f/27.0f);

    // Get current node's value, place into local and shared memory
    curr = d_data[CURRENT];
    s_data[tz][ty][tx] = curr;
    __syncthreads();

    // Get halo regions and place them into shared memory
    
    // Upper midpoints + corners
    if(ty == 1 && iy > 0) {                
        if(tx == 1 && ix > 0) {
            if(tz == 1 && iz > 0)   s_data[tz-1][ty-1][tx-1] = d_data[CURRENT - 1 - N - N*N];
            if(tz == Z && iz < N-1) s_data[tz+1][ty-1][tx-1] = d_data[CURRENT - 1 - N + N*N];
            s_data[tz][ty-1][tx-1] = d_data[CURRENT - 1 - N];
        }
        if(tx == X && ix < N-1) {
            if(tz == 1 && iz > 0)   s_data[tz-1][ty-1][tx+1] = d_data[CURRENT + 1 - N - N*N];
            if(tz == Z && iz < N-1) s_data[tz+1][ty-1][tx+1] = d_data[CURRENT + 1 - N + N*N];
            s_data[tz][ty-1][tx+1] = d_data[CURRENT + 1 - N];
        }
        if(tz == 1 && iz > 0)   s_data[tz-1][ty-1][tx] = d_data[CURRENT - N - N*N];
        if(tz == Z && iz < N-1) s_data[tz+1][ty-1][tx] = d_data[CURRENT - N + N*N];        
        s_data[tz][ty-1][tx] = d_data[CURRENT - N];
    }
    __syncthreads();

    // Lower midpoints + corners
    if(ty == Y && iy < N-1) {
        if(tx == 1 && ix > 0) {
            if(tz == 1 && iz > 0)   s_data[tz-1][ty+1][tx-1] = d_data[CURRENT - 1 + N - N*N];
            if(tz == Z && iz < N-1) s_data[tz+1][ty+1][tx-1] = d_data[CURRENT - 1 + N + N*N];
            s_data[tz][ty+1][tx-1] = d_data[CURRENT - 1 - N];
        }
        if(tx == X && ix < N-1) {
            if(tz == 1 && iz > 0)   s_data[tz-1][ty+1][tx+1] = d_data[CURRENT + 1 + N - N*N];
            if(tz == Z && iz < N-1) s_data[tz+1][ty+1][tx+1] = d_data[CURRENT + 1 + N + N*N];
            s_data[tz][ty+1][tx+1] = d_data[CURRENT + 1 - N];
        }
        if(tz == 1 && iz > 0)   s_data[tz-1][ty+1][tx] = d_data[CURRENT + N - N*N];
        if(tz == Z && iz < N-1) s_data[tz+1][ty+1][tx] = d_data[CURRENT + N + N*N];        
        s_data[tz][ty+1][tx] = d_data[CURRENT + N];
    }
    __syncthreads();

    // Side midpoints
    if(tx == 1 && ix > 0) {
        if(tz == 1 && iz > 0)   s_data[tz-1][ty][tx-1] = d_data[CURRENT - 1 - N*N];
        if(tz == Z && iz < N-1) s_data[tz+1][ty][tx-1] = d_data[CURRENT - 1 + N*N];
        s_data[tz][ty][tx-1] = d_data[CURRENT - 1];
    }
    if(tx == X && ix < N-1) {
        if(tz == 1 && iz > 0)   s_data[tz-1][ty][tx+1] = d_data[CURRENT + 1 - N*N];
        if(tz == Z && iz < N+1) s_data[tz+1][ty][tx+1] = d_data[CURRENT + 1 + N*N];
        s_data[tz][ty][tx+1] = d_data[CURRENT + 1];
    }
    __syncthreads();

    // Front and back halos
    if(tz == 1 && iz > 0)   s_data[tz-1][ty][tx] = d_data[CURRENT - N*N];
    if(tz == Z && iz < N-1) s_data[tz+1][ty][tx] = d_data[CURRENT + N*N];
    __syncthreads();

    // Place node values into local variables

    // Local nodes (1 unit away)
    right = s_data[tz][ty][tx+1]; __syncthreads();    
    left  = s_data[tz][ty][tx-1]; __syncthreads();
    up    = s_data[tz][ty+1][tx]; __syncthreads();
    down  = s_data[tz][ty-1][tx]; __syncthreads();
    front = s_data[tz+1][ty][tx]; __syncthreads();
    back  = s_data[tz-1][ty][tx]; __syncthreads();

    // Midpoints (sqrt(2) units away)
    right_front = s_data[tz+1][ty][tx+1]; __syncthreads();
    right_back  = s_data[tz-1][ty][tx+1]; __syncthreads();
    right_up    = s_data[tz][ty+1][tx+1]; __syncthreads();
    right_down  = s_data[tz][ty-1][tx+1]; __syncthreads();

    left_front  = s_data[tz+1][ty][tx-1]; __syncthreads();
    left_back   = s_data[tz-1][ty][tx-1]; __syncthreads();
    left_up     = s_data[tz][ty+1][tx-1]; __syncthreads();
    left_down   = s_data[tz][ty-1][tx-1]; __syncthreads();

    front_up    = s_data[tz+1][ty+1][tx]; __syncthreads();
    front_down  = s_data[tz+1][ty-1][tx]; __syncthreads();
    back_up     = s_data[tz-1][ty+1][tx]; __syncthreads();
    back_down   = s_data[tz-1][ty+1][tx]; __syncthreads();

    // Corners (sqrt(3) units away)
    right_front_up   = s_data[tz+1][ty+1][tx+1]; __syncthreads();
    right_front_down = s_data[tz+1][ty-1][tx+1]; __syncthreads();
    right_back_up    = s_data[tz-1][ty+1][tx+1]; __syncthreads();
    right_back_down  = s_data[tz-1][ty-1][tx+1]; __syncthreads();

    left_front_up    = s_data[tz+1][ty+1][tx-1]; __syncthreads();
    left_front_down  = s_data[tz+1][ty-1][tx-1]; __syncthreads();
    left_back_up     = s_data[tz-1][ty+1][tx-1]; __syncthreads();
    left_back_down   = s_data[tz-1][ty-1][tx-1]; __syncthreads();
    
    // Don't try to perform calculations on edges of data
    if(ix == 0 || iy == 0 || ix == N-1 || iy == N-1 || iz == 0 || iz == N-1) return;
	
    // Keep input static so that values go up over time. (e.g. constant heat)
    if(ix + iy*N + iz*N*N == MIDDLE) return;

    // Compute output and place into curr and write to smem
    curr += (right + left + up + down + front + back);
    
    curr += (right_front + right_back + right_up + right_down +
                       left_front + left_back + left_up + left_down +
                       front_up + front_down + back_up + back_down);

    curr += (right_front_up + right_front_down + right_back_up + right_back_down +
                       left_front_up + left_front_down + left_back_up + left_back_down);

    curr *= coef;
    
    s_data[tz][ty][tx] = curr;
    __syncthreads();

    //Write data back to global mem
    d_data[CURRENT] = curr;
    __syncthreads();
}

int main(int argc, char* *argv)
{
    if(argc != 3) {printf("USAGE: %s <size> <steps>\n", argv[0]); return 10;} 
    
    const int N     = atoi(argv[1]);    // Data dimensions (N * N * N);
    const int STEPS = atoi(argv[2]);    // Number of iterations to perform.

    // constants to hold grid and threadblock dimensions
    const dim3 blocks ( N/X, N/Y, N/Z );
    const dim3 threads( X, Y, Z );
    
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
