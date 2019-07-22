#ifndef _EDGE_CUDA_CUH_
#define _EDGE_CUDA_CUH_

#include <cuda_runtime.h>
#include <stdio.h>


typedef unsigned int  uint;
typedef unsigned char uchar;



extern "C"
{
	void canny_cuda_impl(uchar*src, uchar *dst, int rows, int cols, float high_thres, float low_thres);

}



#endif // !_EDGE_CUDA_CUH_
