#pragma once
#ifndef GPU_DEFINITIONS_HH
#define GPU_DEFINITIONS_HH

#ifdef __CUDACC__

#define _CUDA_H  __host__
#define _CUDA_D __device__
#define _CUDA_G __global__
#define _CUDA_HD __host__ __device__ 
#else

#define _CUDA_H
#define _CUDA_D
#define _CUDA_G
#define _CUDA_HD

#endif // CUDACC


#endif