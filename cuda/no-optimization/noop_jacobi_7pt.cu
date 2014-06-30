//--------------------------------------------------------------------------------------------
// Author: Adam Barker			email: abarker2@uccs.edu
//
// File: noop_jacobi_7pt.cu		date: June 30, 2014
//
// This program performs a simple averaging of node values using a 7-point 3D Jacobi stencil.
// This program also incorporates the use of shared memory to speed up memory accesses and
// staggers reads of the halo regions so that race conditions do not exist among threads.
//
// This program contains no advanced optimizations.
//--------------------------------------------------------------------------------------------
#include <stdio.h>
#include <output.c>

__global__ void kernel(float * d_data)
{
    // Global data location
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int iz = threadIdx.z + blockDim.z * blockIdx.z;
    
    // local data location
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    // Shared Memory dimensions
    int bx = blockDim.x + 2;
    int by = blockDim.y + 2;
    int bz = blockDim.z + 2;

    // Global Memory dimensions
    int dimx = blockDim.x * gridDim.x;
    int dimy = blockDim.y * gridDim.y;
    int dimz = blockDim.z * gridDim.z;

    // Shared and Global memory node location constants
    int CURRENT_G = ix + iy*dimx + iz*dimx*dimy;
    int CURRENT_S = tx + ty*bx + tz*bx*by;

    // Dynamic shared memory declaration
    extern __shared__ float s_data[];

    // Node variables
    float curr;
    float right;
    float left;
    float up;
    float down;
    float front;
    float back;

    // constant to average node values
    const float coef = 1.0f / 7.0f;
    
    // get node value from global memory
    curr = d_data[CURRENT_G];
    __syncthreads();

    // place into smem
    s_data[CURRENT_S] = curr;
    __syncthreads();
    
    // Load halo regions into smem
    if((ix == 0 || iy == 0 || iz == 0) || (ix >= dimx-1 || iy >= dimy-1 || iz >= dimz-1)) return;
    if(CURRENT_G == dimx/2 + dimx*(dimy/2) + dimx*dimy*(dimz/2)) return;

    if(tx == 1)  s_data[CURRENT_S - 1] = d_data[CURRENT_G - 1];
    __syncthreads();
    if(tx == bx-2) s_data[CURRENT_S + 1] = d_data[CURRENT_G + 1];
    __syncthreads();

    if(ty == 1)  s_data[CURRENT_S - bx] = d_data[CURRENT_G - dimx];
    __syncthreads();
    if(ty == by-2) s_data[CURRENT_S + bx] = d_data[CURRENT_G + dimx];
    __syncthreads();

    if(tz == 1)  s_data[CURRENT_S - bx*by] = d_data[CURRENT_G - dimx*dimy];
    __syncthreads();
    if(tz == bz-2) s_data[CURRENT_S + bx*by] = d_data[CURRENT_G + dimx*dimy];
    __syncthreads();

    // get node values from smem
    right = s_data[CURRENT_S + 1]; __syncthreads();
    left  = s_data[CURRENT_S - 1]; __syncthreads();
    up    = s_data[CURRENT_S + bx]; __syncthreads();
    down  = s_data[CURRENT_S - bx]; __syncthreads();
    front = s_data[CURRENT_S + bx*by]; __syncthreads();
    back  = s_data[CURRENT_S - bx*by]; __syncthreads();

    // compute output
    curr = coef * (curr + right + left + up + down + front + back);
    __syncthreads();

    // place output into global memory
    d_data[CURRENT_G] = curr;
    __syncthreads();
}

int main(int argc, char* *argv)
{
    if(argc != 8) {printf("USAGE: %s <bx> <by> <bz> <tx> <ty> <tz> <steps>\n", argv[0]); return 10;}

    // block sizes, thread sizes, and number of steps from command line args
    const int bx    = atoi(argv[1]);
    const int by    = atoi(argv[2]);
    const int bz    = atoi(argv[3]);
    const int tx    = atoi(argv[4]);
    const int ty    = atoi(argv[5]);
    const int tz    = atoi(argv[6]);
    const int STEPS = atoi(argv[7]);

    // data dimensions
    const int X = bx * tx;
    const int Y = by * ty;
    const int Z = bz * tz;
    
    // Size of array in bytes and amount of dynamic smem to allocate
    const int ARRAY_BYTES  = bx*tx * by*ty * bz*tz * sizeof(float);
    const int SHARED_BYTES = (tx+2) * (ty+2) * (tz+2) * sizeof(float);

    // constants for block and thread dimensions
    const dim3 blocks (bx, by, bz);
    const dim3 threads(tx, ty, tz);

    // host and device array declarations
    float * h_data;
    float * d_data;

    // host and device array allocations
    h_data = (float*)malloc(ARRAY_BYTES);
    cudaMalloc(&d_data, ARRAY_BYTES);

    // host array initialization
    for(int k=0; k<Z; k++) {
        for(int j=0; j<Y; j++) {
            for(int i=0; i<X; i++) {
                h_data[i + j*X + k*X*Y] = 5.0f;
            }
        }
    }

    // set middle node to higher value for actual compuations to take place
    h_data[X/2 + X*(Y/2) + X*Y*(Z/2)] = 10.0f;

    // copy array to GPU
    cudaMemcpy(d_data, h_data, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // call kernel number of times specified
    for(int step=0; step < STEPS; step++)
    kernel<<< blocks, threads, SHARED_BYTES >>>( d_data);

    // copy array from GPU back to host
    cudaMemcpy(h_data, d_data, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // output data
    output("out.image", h_data, X, Y, Z);

    free(h_data);
    cudaFree(d_data);
    
    return 0;
}
