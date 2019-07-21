#pragma once
#include <cuda_runtime.h>

#include <stdio.h>

typedef unsigned int  uint;
typedef unsigned char uchar;



// Marco for error checking
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
//{
//	if (code != cudaSuccess)
//	{
//		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//	}
//}


extern "C"
{
	void canny_cuda_impl(
		uchar*src, uchar *dst,
		int rows,		 int cols,
		float high_thres, float low_thres);


	void canny_cuda_init();
	
}
