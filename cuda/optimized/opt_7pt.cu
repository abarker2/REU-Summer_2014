#include <stdio.h>
#include <output.c>

#define INIT  5.0f
#define START 10.0f

__global__ void stencil(float * d_data, int dx, int dy, int bx, int by, int dz)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = 1;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = 1;

    int pos_gl = ix + iy*dx + iz*dx*dy;
    int pos_sh = tx + ty*bx + tz*bx*by;

    extern __shared__ float s_data[];

    float curr;
    float right, left;
    float up, down;
    float front, back;

    const float coef = 1.0f / 7.0f;

    s_data[pos_sh] = d_data[pos_gl];
    __syncthreads();

    s_data[pos_sh - bx*by] = d_data[pos_gl - dx*dy];
    __syncthreads();

    s_data[pos_sh + bx*by] = d_data[pos_gl + dx*dy];
    __syncthreads();

    if(tx == 1 && ix > 0) {
        s_data[pos_sh - 1] = d_data[pos_gl - 1];
        s_data[pos_sh + bx*by - 1] = d_data[pos_gl + dx*dy - 1];
    }
    __syncthreads();

    if(tx == bx-2 && ix < dx-1) {
        s_data[pos_sh + 1] = d_data[pos_gl + 1];
        s_data[pos_sh + bx*by + 1] = d_data[pos_gl + bx*by + 1];
    }
    __syncthreads();

    if(ty == 1 && iy > 0) {
        s_data[pos_sh - bx] = d_data[pos_gl - dx];
        s_data[pos_sh + bx*by - bx] = d_data[pos_gl + dx*dy - dx];
    }
    __syncthreads();

    if(ty == by-2 && iy < dy-1) {
        s_data[pos_sh + bx] = d_data[pos_gl + dx];
        s_data[pos_sh + bx*by + bx] = d_data[pos_gl + dx*dy + dx];
    }
    __syncthreads();

    if(!(ix==0 || ix==dx-1 || iy==0 || iy==dy-1)) {
        curr  = s_data[pos_sh];         __syncthreads();        
        right = s_data[pos_sh + 1];     __syncthreads();
        left  = s_data[pos_sh - 1];     __syncthreads();
        up    = s_data[pos_sh - bx];    __syncthreads();
        down  = s_data[pos_sh + bx];    __syncthreads();
        front = s_data[pos_sh + bx*by]; __syncthreads();
        back  = s_data[pos_sh - bx*by]; __syncthreads();

        curr  = coef * (curr + right + left + up + down + front + back);
        __syncthreads();

        d_data[pos_gl] = curr;
        __syncthreads();
    }

    for(pos_gl; pos_gl < dx*dy*dz; pos_gl += dx*dy) {
        s_data[pos_sh - bx*by] = s_data[pos_sh];
        __syncthreads();

        s_data[pos_sh] = s_data[pos_gl + bx*by];
        __syncthreads();

        s_data[pos_sh + bx*by] = d_data[pos_gl + dx*dy];
        __syncthreads();

        if(tx == 1 && ix > 0) {
            s_data[pos_sh - 1] = d_data[pos_gl - 1];
            s_data[pos_sh + bx*by - 1] = d_data[pos_gl + dx*dy - 1];
        }
        __syncthreads();

        if(tx == bx-2 && ix < dx-1) {
            s_data[pos_sh + 1] = d_data[pos_gl + 1];
            s_data[pos_sh + bx*by + 1] = d_data[pos_gl + bx*by + 1];
        }
        __syncthreads();

        if(ty == 1 && iy > 0) {
            s_data[pos_sh - bx] = d_data[pos_gl - dx];
            s_data[pos_sh + bx*by - bx] = d_data[pos_gl + dx*dy - dx];
        }
        __syncthreads();

        if(ty == by-2 && iy < dy-1) {
            s_data[pos_sh + bx] = d_data[pos_gl + dx];
            s_data[pos_sh + bx*by + bx] = d_data[pos_gl + dx*dy + dx];
        }
        __syncthreads();

        if(!(ix==0 || ix==dx-1 || iy==0 || iy==dy-1)) {
            curr  = s_data[pos_sh];         __syncthreads();        
            right = s_data[pos_sh + 1];     __syncthreads();
            left  = s_data[pos_sh - 1];     __syncthreads();
            up    = s_data[pos_sh - bx];    __syncthreads();
            down  = s_data[pos_sh + bx];    __syncthreads();
            front = s_data[pos_sh + bx*by]; __syncthreads();
            back  = s_data[pos_sh - bx*by]; __syncthreads();

            curr  = coef * (curr + right + left + up + down + front + back);
            __syncthreads();

            d_data[pos_gl] = curr;
            __syncthreads();
        }
    }
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

int main(int argc, char* *argv)
{
    if(argc != 7) {printf("USAGE: %s <bx> <by> <tx> <ty> <Z> <steps>\n", argv[0]); return 10;}

    int bx = atoi(argv[1]), by = atoi(argv[2]);
    int tx = atoi(argv[3]), ty = atoi(argv[4]);
    int Z  = atoi(argv[5]);
    const int STEPS = atoi(argv[6]);

    const dim3 blocks (bx, by, 1);
    const dim3 threads(tx, ty, 1);

    const int ARRAY_BYTES  = bx*tx * by*ty * Z * sizeof(float);
    const int SHARED_BYTES = (tx+2) * (ty+2) * (3) * sizeof(float);

    float * d_data;
    float * h_data;

    cudaMalloc(&d_data,  ARRAY_BYTES);
    h_data = (float*)malloc(ARRAY_BYTES);

    initialize<<<dim3(bx, by, Z), threads>>>(d_data, bx*tx, by*ty, Z);

    for(int step=0; step < STEPS; step+=2) {
        stencil<<<blocks, threads, SHARED_BYTES>>>(d_data, bx*tx, by*ty, tx+2, ty+2, Z);
    }

    cudaMemcpy(h_data, d_data, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    output("out.image", h_data, bx*tx, by*ty, Z);

    free(h_data);
    cudaFree(d_data);

    return 0;
}
