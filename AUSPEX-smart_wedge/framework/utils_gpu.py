import numpy as np
import pycuda.autoinit
import pycuda.cumath as cm
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culinalg
import skcuda.misc as misc
from pycuda.compiler import SourceModule as SM


culinalg.init()
misc.init()

MAX_THREADS_PER_BLOCK = drv.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)

BLOCK_SIZE = int(np.sqrt(MAX_THREADS_PER_BLOCK))
FLOAT_SIZE = np.float32

sq_sum_codetext = """
__global__ void sq_sum(float *dest, float *dest2, float *a, float *b, int* SZ)
{
    // Store parameters
    const int M = SZ[0];
    const int N = SZ[1];
    const int M2 = SZ[2];
    const int t = threadIdx.x;
    const int TBP = blockDim.x;
    const int bx = blockIdx.x;
    const int GSZ = gridDim.x;
    // Initialize iterators
    int i,I,j,inv;
    // Initialize 2D shaped memory. The first axis stores the summations for
    // each row. The second axis is the column and data gets reduced along it.
    __shared__ float aS[BLOCK_SIZE][GRID_SIZE];
    // Loop to square sum-reduce each row of first array
    for(i=0;i<M/GSZ;i++)
    {
        j = bx;
        I = i*GSZ+bx;
        inv = TBP;
        aS[t][j] = a[I*N+t]*a[I*N+t];
        __syncthreads();
        if(t+inv<N)
            aS[t][j] += a[I*N+t+inv]*a[I*N+t+inv];
        __syncthreads();
        inv = inv/2;
        while(inv!=0)
        {
            if(t<inv)
                aS[t][j] += aS[t+inv][j];
            __syncthreads();
            inv = inv/2;
        }
        __syncthreads();
        if(t==0)
            dest[I] = aS[0][j];
        __syncthreads();
    }
    // Loop to square sum-reduce each row of second array
    for(i=0;i<M2/GSZ;i++)
    {
        j = bx;
        I = i*GSZ+bx;
        inv = TBP;
        aS[t][j] = b[I*N+t]*b[I*N+t];
        __syncthreads();
        if(t+inv<N)
            aS[t][j] += b[I*N+t+inv]*b[I*N+t+inv];
        __syncthreads();
        inv = inv/2;
        while(inv!=0)
        {
            if(t<inv)
                aS[t][j] += aS[t+inv][j];
            __syncthreads();
            inv = inv/2;
        }
        __syncthreads();
        if(t==0)
            dest2[I] = aS[0][j];
        __syncthreads();
    }
}
"""
addvecs_codetext = """
__global__ void add_vectors_broadcast(float *dest, float *a, float *b, int* SZ)
{
    const int M = SZ[0];
    const int N = SZ[1];
    const int S = SZ[2];
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int BSZ = blockDim.x;
    int t;
    for (int s=0;s<S;s++)
    {
        t = s*BSZ+tx;
        if(t<N)
            dest[bx*N+t] = b[t] + a[bx];
        __syncthreads();
    }
}
"""

addvecs_bcast_gpu = SM(addvecs_codetext).get_function("add_vectors_broadcast")
BSZ, GSZ = BLOCK_SIZE, BLOCK_SIZE
block_blocksize_define_str = "#define BLOCK_SIZE " + str(BSZ)
block_gridsize_define_str = "#define GRID_SIZE " + str(GSZ)
define_str = "\n".join([block_blocksize_define_str, block_gridsize_define_str])
sq_sum_gpu = SM(define_str + sq_sum_codetext).get_function("sq_sum")


def addvecs_gpu(a_gpu, b_gpu):
    M, N = a_gpu.shape[0], b_gpu.shape[0]
    out_gpu = gpuarray.empty((M, N), dtype=FLOAT_SIZE)
    BSZ = min(MAX_THREADS_PER_BLOCK, N)
    GSZ = M
    num_iter = int(np.ceil(N/float(MAX_THREADS_PER_BLOCK)))
    a_shp = np.int32([M, N, num_iter])
    addvecs_bcast_gpu(out_gpu, a_gpu, b_gpu, drv.In(a_shp),
                      block=(BSZ, 1, 1), grid=(GSZ, 1))
    return out_gpu


def sq_sums(a_gpu, b_gpu, GSZ=GSZ):
    M, N, R = a_gpu.shape[0], b_gpu.shape[0], a_gpu.shape[1]
    BSZ = 2**int(np.ceil(np.log(R)/np.log(2))-1)
    out_gpu1 = gpuarray.empty(M, dtype=FLOAT_SIZE)
    out_gpu2 = gpuarray.empty(N, dtype=FLOAT_SIZE)

    shp = np.int32([M, R, N])
    sq_sum_gpu(out_gpu1, out_gpu2, a_gpu, b_gpu, drv.In(shp), block=(BSZ, 1, 1), grid=(GSZ, 1))
    out_gpu = addvecs_gpu(out_gpu1, out_gpu2)
    return out_gpu


def cdist_gpu(a, b):
    c_gpu = sq_sums(a, b)
    return cm.sqrt(culinalg.add_dot(a, b, c_gpu, transb='T', alpha=-2.0))


sinc_codetext = """
#include <math.h>
__global__ void sinc(float *a, int* SZ)
{
    const int M = SZ[0];
    const int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t<=M){
        if (a[t]==0.0){
            a[t] = 1.0;
        }
        else{
            a[t] = sin(M_PI * a[t]) / (M_PI * a[t]); 
        }
    }
    __syncthreads();
}
"""

sinc_gpu = SM(sinc_codetext).get_function("sinc")


def sinc(x, y=None):
    shape = x.shape
    x = x.ravel().astype(np.float32)
    m = x.shape[0]
    if y is None:
        y = gpuarray.ones_like(x, np.float32)
    else:
        assert isinstance(y, gpuarray.GPUArray)
    BSZ = (MAX_THREADS_PER_BLOCK, 1, 1)
    GSZ = (m//MAX_THREADS_PER_BLOCK + 1, 1)
    sinc_gpu(y, x, drv.In(np.asarray([m])), block=BSZ, grid=GSZ)
    return y.reshape(shape)


## Aqui dá pra mudar a saída pra float2 e obter os angulos na direcão passiva tambem
atan2_codetext = """
#include <math.h>
__global__ void arctan2(float *out, float3 *x, int *SZ)
{
    const int M = SZ[0];
    const int t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t<=M){
        out[t] = atan2(x[t].x, x[t].z);
    }
    __syncthreads();
}
"""

atan2_gpu = SM(atan2_codetext).get_function("arctan2")


def atan2(x):
    assert isinstance(x, gpuarray.GPUArray)
    m = x.size//3
    out = gpuarray.empty(m, dtype=x.dtype)
    shape = x.shape[0:2]
    BSZ = (min(m, MAX_THREADS_PER_BLOCK), 1, 1)
    GSZ = (m//MAX_THREADS_PER_BLOCK + 1, 1)
    atan2_gpu(out, x.reshape(-1, 3, order="C"), drv.In(np.asarray([m])), block=BSZ, grid=GSZ)
    return out.reshape(shape)


def directivity_weights(elem_coords, img_coords, ka):
    assert img_coords.shape[1] == elem_coords.shape[1]
    assert isinstance(img_coords, gpuarray.GPUArray)
    assert isinstance(elem_coords, gpuarray.GPUArray)
    elem_coords = -1*elem_coords
    df = add3darrs_gpu(img_coords, elem_coords)
    dg = atan2(df)
    return sinc(ka / 2 * cm.sin(dg)) * cm.cos(dg)


addarrs_codetext = """
__global__ void add_3darrs_broadcast(float3 *dest, float3 *a, float3 *b, int* SZ)
{
    const int M = SZ[0];
    const int N = SZ[1];
    const int S = SZ[2];
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int BSZ = blockDim.x;
    int t;
    for (int s=0;s<S;s++)
    {
        t = s*BSZ+tx;
        if(t<N)
            dest[bx*N+t].x = b[t].x + a[bx].x;
            dest[bx*N+t].y = b[t].y + a[bx].y;
            dest[bx*N+t].z = b[t].z + a[bx].z;
        __syncthreads();
    }
}
"""
addarrs_bcast_gpu = SM(addarrs_codetext).get_function("add_3darrs_broadcast")


def add3darrs_gpu(a_gpu, b_gpu):
    assert isinstance(a_gpu, gpuarray.GPUArray)
    assert isinstance(b_gpu, gpuarray.GPUArray)
    M, N = a_gpu.shape[0], b_gpu.shape[0]
    out_gpu = gpuarray.empty((M, N, 3), dtype=FLOAT_SIZE)
    BSZ = min(MAX_THREADS_PER_BLOCK, N)
    GSZ = M
    num_iter = int(np.ceil(N/float(MAX_THREADS_PER_BLOCK)))
    a_shp = np.int32([M, N, num_iter])
    addarrs_bcast_gpu(out_gpu, a_gpu, b_gpu, drv.In(a_shp),
                      block=(BSZ, 1, 1), grid=(GSZ, 1))
    return out_gpu
