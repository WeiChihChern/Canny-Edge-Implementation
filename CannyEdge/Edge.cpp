
#include "DefineFlags.hpp"
#include "Edge.h"

#ifdef USE_CUDA
#include "Edge_Cuda.cuh"
#endif

#include <algorithm>
#include <math.h>



#ifdef USE_CUDA
void Edge::cannyEdge_cuda(Mat& src, Mat& dst, const float& high_thres, const float& low_thres)
{
	canny_cuda_impl(src.data, dst.data, src.rows, src.cols, high_thres, low_thres);

	this->hysteresis_threshold(dst);

	return;
}
#endif



























