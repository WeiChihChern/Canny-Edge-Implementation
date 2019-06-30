#include "Edge.h"

#include <algorithm>
#define _USE_MATH_DEFINES // for C++
#include <math.h>




Edge::Edge()
{
}


Edge::~Edge()
{
}









// void Edge::CannyEdge(Mat& src, Mat &dst, float high_thres, float low_thres) {

// 	this->rows = src.rows;
// 	this->cols = src.cols;
// 	this->size = this->rows * this->cols;

// 	Mat copy1, copy2;
// 	this->conv2<uchar, short>(src, copy1, sobel_horizontal);
// 	this->conv2<uchar, short>(src, copy2, sobel_vertical);

// 	this->calculate_Magnitude<short,short>(copy1, copy2, this->magnitude, true);
// 	this->calculate_Gradients<short, short>(copy1, copy2, this->gradient);
	


// 	this->nonMaxSuppresion(magnitude, gradient, copy1, copy2, high_thres, low_thres);



// 	dst = this->hysteresis_threshold(suppressed);

// 	this->release();
	
// 	return;
// }












void Edge::cannyEdge2(Mat& src, Mat&dst, float high_thres, float low_thres) {

	this->rows = src.rows;
	this->cols = src.cols;
	this->size = this->rows * this->cols;


#ifdef _OPENMP
	omp_set_num_threads(threadControl(this->size));
#endif


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

	

	// Save magnitude result in unsigned char (uchar) 
	this->calculate_Magnitude<short, short>(gx, gy, this->magnitude, true);

	// Save gradient result in signed char (schar)
	this->calculate_Gradients<short, short>(gx, gy, this->gradient);


	this->nonMaxSuppresion(this->magnitude, this->gradient, gy, gx, this->suppressed, high_thres, low_thres);

	dst = this->hysteresis_threshold(suppressed);

	this->release();

	return;

}












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



	#pragma omp parallel for 
	for (size_t i = 2; i < this->rows-2; ++i) {
		dst_ptr = dst.ptr<uchar>(i);
		mag_ptr = magnitude.ptr<uchar>(i);
		gra_ptr = gradient.ptr<schar>(i);
		gx_p    = gx.ptr<short>(i);
		gy_p    = gy.ptr<short>(i);

#ifdef __GNUC__
		#pragma omp simd
#endif
		for (size_t j = 2; j < this->cols-2; ++j) 
		{
			cur_mag_val = *(mag_ptr+j);
			theta       = gra_ptr[j];
		

			if ( cur_mag_val > low_thres && cur_mag_val != 0 ) // Edge pixel
			{ 
				if (theta == 90) 
				{
					// vertical direction
					if ( cur_mag_val > mag_ptr[j - cols] && cur_mag_val >= mag_ptr[j + cols] ) 
						dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : cur_mag_val;
				}
				else if (theta == 0) 
				{
					// horizontal direction
					if (cur_mag_val > mag_ptr[j - 1] && cur_mag_val >= mag_ptr[j + 1]) 
						dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : cur_mag_val;
				}
				else  // bottom-left to top-right  or  bottom-right to top-left direction
				{ 
					int d = (gy_p[j] * gx_p[j] < 0) ? 1 : -1;
					if (cur_mag_val >= mag_ptr[j + cols - d] && cur_mag_val > mag_ptr[j - cols + d]) 
						dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : cur_mag_val;
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
}













Mat Edge::hysteresis_threshold(const Mat& src) {

	if (src.empty() || src.channels() == 3) { cout << "hysteresis_threshold() error!\n"; return Mat(0,0,CV_8UC1); }

	uchar* dst_p;
	const uchar* nonM_p;
	Mat dst(this->rows, this->cols, CV_8UC1); 


	// In nonMax(), not only find the max along the graident direction,
	// but everything >= high_thres is set to 255, and everything < low_thres
	// is set to zero. 
	// Check the pixel value between high_thres & lower_thres to see if there's any 
	// strong pixel in the 8-neighbor, and set itself to 255 if it does.
	
#pragma omp parallel for
	for (size_t i = 1; i < this->rows-1; ++i)
	{
		dst_p  = dst.ptr<uchar>(i); 
		nonM_p = src.ptr<uchar>(i); // non max result pointer
			
#ifdef __GNUC__
		#pragma omp simd // for -O2 optimization
#endif	
		for (size_t j = 1; j < this->cols-1; ++j) 
		{
			uchar val = nonM_p[j];
			if(val == 0)
				dst_p[j] = val;
			else if (val == 255)
				dst_p[j] = val;
			else
			{
				if (*(nonM_p + j - 1) == 255 || *(nonM_p + j + 1) == 255 || *(nonM_p+j - cols) == 255 || 
				    *(nonM_p + j + cols) == 255 || *(nonM_p + j - cols - 1) == 255 || 
					*(nonM_p + j - cols + 1) == 255 || *(nonM_p + j + cols + 1) == 255 || *(nonM_p + j + cols - 1) == 255) 
				{
					dst_p[j] = 255;
				}
				else // No strong pixel (=255) in 8 neighbors
				{ 
					dst_p[j] = 0;
				}
			}		
		}
	}

	this->edge2zero<uchar>(dst);

#ifdef DEBUG_IMSHOW_RESULT
	imshow("Hysteresis threshold result", dst);
	waitKey(10);
#endif 



	return dst;
}



