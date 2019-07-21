#include "Edge.h"

#include <algorithm>
#define _USE_MATH_DEFINES // for C++
#include <math.h>

#include "Edge_Cuda.cuh"



Edge::Edge()
{
}


Edge::~Edge()
{
}








void Edge::cannyEdge_cuda(Mat& src, Mat& dst, const float& high_thres, const float& low_thres)
{
	canny_cuda_impl(src.data, dst.data, src.rows, src.cols, high_thres, low_thres);

	this->hysteresis_threshold(dst);

	return;
}



















void Edge::cannyEdge2(Mat& src, Mat&dst, float high_thres, float low_thres) {


	this->rows = src.rows;
	this->cols = src.cols;
	this->size = this->rows * this->cols;



	Mat gx(src.rows, src.cols, CV_16SC1); // Short type
	this->conv2_h_sobel<uchar, short>(       src, gx, this->sobel_one);
	this->conv2_v_sobel<short, short>(gx.clone(), gx, this->sobel_two);

	
	Mat gy(src.rows, src.cols, CV_16SC1); // Short type
	this->conv2_h_sobel<uchar, short>(       src, gy, this->sobel_two);
	this->conv2_v_sobel<short, short>(gy.clone(), gy, this->sobel_one);


#ifdef DEBUG_IMSHOW_RESULT
	Mat gy_show, gx_show;
	gy.convertTo(gy_show, CV_8UC1);
	gx.convertTo(gx_show, CV_8UC1);
	imshow("conv2_sobel() G(y) in 8-bit (from float)", gy_show);
	imshow("conv2_sobel() G(x) in 8-bit (from float)", gx_show);
	waitKey(10);
#endif 

#ifdef _OPENMP
	omp_set_num_threads(threadControl(this->size));
#endif
	

	// Save magnitude result in unsigned char (uchar) 
	this->calculate_Magnitude<short, short>(gx, gy, this->magnitude, true);

	// Save gradient result in signed char (schar)
	this->calculate_Gradients<short, short>(gx, gy, this->gradient);


	this->nonMaxSuppresion(this->magnitude, this->gradient, gy, gx, this->suppressed, high_thres, low_thres);

	this->hysteresis_threshold(this->suppressed);

	dst = this->suppressed;

	this->release();

	return;
} // end of cannyedge2












void Edge::nonMaxSuppresion(
	const Mat &magnitude, const Mat &gradient, 
	const Mat& gy, const Mat& gx, Mat &dst, 
	float high_thres, float low_thres) 
	{
	
	// Both magnitude & gradient are in float type
	if(dst.empty()) dst = Mat (this->rows, this->cols, CV_8UC1, Scalar(0)); 

	uchar* dst_ptr;
	const uchar* mag_ptr;
	const schar* gra_ptr;

	short theta;
	const short *gx_p, *gy_p;
	uchar cur_mag_val;



	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 2; i < this->rows-2; ++i) {
		dst_ptr = dst.ptr<uchar>(i);
		mag_ptr = magnitude.ptr<uchar>(i);
		gra_ptr = gradient.ptr<schar>(i);
		gx_p    = gx.ptr<short>(i);
		gy_p    = gy.ptr<short>(i);

#ifdef __GNUC__
		#pragma omp simd
#endif
		for (int j = 2; j < this->cols-2; ++j)
		{
				cur_mag_val = *(mag_ptr+j);
				theta       = gra_ptr[j];
			

				if ( cur_mag_val > low_thres && cur_mag_val != 0 ) // Edge pixel
				{ 
						if (theta == 90) 
						{
							// vertical direction
								if ( cur_mag_val > mag_ptr[j - cols] && cur_mag_val >= mag_ptr[j + cols] ) 
								{
										if(cur_mag_val >= high_thres) dst_ptr[j] = 255;
										else 						  dst_ptr[j] = 125;
								}
									
						}
						else if (theta == 0) 
						{
								// horizontal direction
								if (cur_mag_val > mag_ptr[j - 1] && cur_mag_val >= mag_ptr[j + 1]) 
								{
										if(cur_mag_val >= high_thres) dst_ptr[j] = 255;
										else 						  dst_ptr[j] = 125;
								}
									
						}
						else  // bottom-left to top-right  or  bottom-right to top-left direction
						{ 
								int d = (gy_p[j] * gx_p[j] < 0) ? 1 : -1;
								if (cur_mag_val >= mag_ptr[j + cols - d] && cur_mag_val > mag_ptr[j - cols + d]) 
								{
										if(cur_mag_val >= high_thres) dst_ptr[j] = 255;
										else 						  dst_ptr[j] = 125;
								}
									
						}
				} 
				else // Non edge pixel
					dst_ptr[j] = 0;
		}
	}


#ifdef DEBUG_IMSHOW_RESULT
	imshow("Non maximum suppression result", dst);
	waitKey(10);
#endif 

	return;
}  // end of nonMax









void Edge::hysteresis_threshold(Mat& src) 
{
	uchar *img_start = src.ptr<uchar>(0);


 #pragma omp parallel for //schedule(dynamic,1)
	for (int i = 2; i < src.rows-1; i++)
	{
		uchar* img_p = src.ptr<uchar>(i);
		
#ifdef __GNUC__
		#pragma omp simd  
#endif	
		for (int j = 2; j < src.cols-1; j++)
		{
			if(img_p[j] == 125) 
			{
				// bool b = canny_hysteresis_dfs(img_start, i, j, src.rows, src.cols, 0);
				// if(!b) img_p[j] = 0;
				if( !(canny_hysteresis_dfs(img_start, i, j, src.rows, src.cols, 0)) ) img_p[j] = 0;
			}
		}
	}



}



