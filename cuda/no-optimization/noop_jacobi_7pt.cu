//--------------------------------------------------------------------------------------------
// Author: Adam Barker			email: abarker2@uccs.edu
//
// File: noop_jacobi_7pt.cu		date: July 1, 2014
//
// This program performs a simple averaging of node values using a 7-point 3D Jacobi stencil.
// This program also incorporates the use of shared memory to speed up memory accesses and
// staggers reads of the halo regions so that race conditions do not exist among threads.
//
// This program contains no advanced optimizations.
//--------------------------------------------------------------------------------------------
#include <stdio.h>
#include <output.c>

#define INIT  5.0f
#define START 10.0f

///////////////////////////////////////////////////////////////
/// This function is the actual stencil kernel that         ///
/// performs an averaging 7 point stencil on the input data ///
///////////////////////////////////////////////////////////////
__global__
void stencil(float * d_data, const int dx, const int dy, const int dz,  // global data dimensions
                             const int bx, const int by, const int bz ) // shared data dimensions
{
    // Global data coordinates
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int iz = threadIdx.z + blockIdx.z * blockDim.z;

    // Local data coordinates with halo radius added
    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;
    const int tz = threadIdx.z + 1;

    // global and shared memory location constants
    const int CURRENT_G = ix + iy*dx + iz*dx*dy;
    const int CURRENT_S = tx + ty*bx + tz*bx*by;

    // Dynamic shared memory declaration
    extern __shared__ float s_data[];

    // local node variable declarations
    float curr;     // Current node
    float right;    // node right of current node
    float left;     // node left of current node
    float up;       // node above current node
    float down;     // node below current node
    float front;    // node in front of current node
    float back;     // node behind current node

    // number to multiply nodes by (average)
    const float coef = 1.0f/7.0f;

    curr = d_data[CURRENT_G];   // fetch current node value from global memory
    s_data[CURRENT_S] = curr;   // place into shared memory
    __syncthreads();
    
    // Don't perform calculations on edge nodes or the middle node
    if( (ix == 0 || ix == dx-1 || iy == 0 || iy == dy-1 || iz == 0 || iz == dz-1)
        || (CURRENT_G == dx/2 + dx*(dy/2) + dx*dy*(dz/2)) ) return;

    /*******************************
     * Load halo regions into smem *
     *******************************/

    // halo region to the left and right of this block
    if(tx == 1)    s_data[CURRENT_S - 1] = d_data[CURRENT_G - 1];
    if(tx == bx-2) s_data[CURRENT_S + 1] = d_data[CURRENT_G + 1];
    __syncthreads();

    // halo region above and below this block
    if(ty == 1)    s_data[CURRENT_S - bx] = d_data[CURRENT_G - dx];
    if(ty == by-2) s_data[CURRENT_S + bx] = d_data[CURRENT_G + dx];
    __syncthreads();

    // halo region behind and in front of this block
    if(tz == 1)    s_data[CURRENT_S - bx*by] = d_data[CURRENT_G - dx*dy];
    if(tz == bz-2) s_data[CURRENT_S + bx*by] = d_data[CURRENT_G + dx*dy];
    __syncthreads();

    /**********************************
     * retrieve node values from smem *
     **********************************/
    right = s_data[CURRENT_S + 1];      __syncthreads();
    left  = s_data[CURRENT_S - 1];      __syncthreads();
    up    = s_data[CURRENT_S - bx];     __syncthreads();
    down  = s_data[CURRENT_S + bx];     __syncthreads();
    front = s_data[CURRENT_S + bx*by];  __syncthreads();
    back  = s_data[CURRENT_S - bx*by];  __syncthreads();

    /**********************
     * Perform compuation *
     **********************/
    curr = coef * (coef + right + left + up + down + front + back);
    __syncthreads();

    // Write result back to global memory
    d_data[CURRENT_G] = curr;
    __syncthreads();
}

///////////////////////////////////////////////////////////////
/// This function initializes an array with set values in a ///
/// parallel fashion for speed up over CPU initialization   ///     
///////////////////////////////////////////////////////////////
__global__
void initialize(float * d_data, const int dx, const int dy, const int dz)
{
    // Global coordinates
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    // Write location
    int CURRENT_G = ix + iy*dx + iz*dx*dy;

    // place initial value into data
    d_data[CURRENT_G] = INIT;

    // if at the middle of the array, write starting value
    if(CURRENT_G = dx/2 + dx*(dy/2) + dx*dy*(dz/2))
        d_data[CURRENT_G] = START;
}

///////////////////////////////////////////////////////////////
/// This is the main function which handles argument        ///
/// parsing and kernel launching.                           ///
///////////////////////////////////////////////////////////////
int main(int argc, char* *argv)
{
    if(argc != 8) {printf("USAGE: %s <bx> <by> <bz> <tx> <ty> <tz> <steps>\n", argv[0]); return 10;}

    // set constants from command line arguments
    const int bx = atoi(argv[1]);
    const int by = atoi(argv[2]);
    const int bz = atoi(argv[3]);

    const int tx = atoi(argv[4]) + 2;
    const int ty = atoi(argv[5]) + 2;
    const int tz = atoi(argv[6]) + 2;

    const int STEPS = atoi(argv[7]);

    const int dx = bx*(tx-2);
    const int dy = by*(ty-2);
    const int dz = bz*(tz-2);

    // number of blocks and threads per block for kernel execution
    const dim3 blocks (bx, by, bz);
    const dim3 threads(tx-2, ty-2, tz-2);

    // Array size and shared mem size declarations
    const int ARRAY_BYTES  = dx * dy * dz * sizeof(float);
    const int SHARED_BYTES = tx * ty * tz * sizeof(float);

    // Host and device array declarations & allocations
    float * h_data;
    float * d_data;

	printf("DATA DIMENSIONS: %d x %d x %d\n", dx, dy, dz);

    h_data = (float*)malloc(ARRAY_BYTES);
    cudaMalloc(&d_data, ARRAY_BYTES);

    initialize<<<blocks, threads>>>(d_data, dx, dy, dz);

    for(int step=0; step < STEPS; step++)
        stencil<<<blocks, threads, SHARED_BYTES>>>(d_data, dx, dy, dz, tx, ty, tz);

    cudaMemcpy(h_data, d_data, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    output("out.image", h_data, dx, dy, dz);

    free(h_data);
    cudaFree(d_data);

    cudaDeviceReset();

    return 0;
}
