#pragma once


#include <vector>
#include <thread>

#include "opencv2/opencv.hpp"
#include "Utils.h"
#include "DefineFlags.hpp"



using namespace std;
using namespace cv;


constexpr auto OFFSET   = 1;      
typedef  unsigned char  uchar;



class Edge : public Utils // Utils includes 2-D & 1-D convolution functions
{
public:
	// Two sobel 2-D kernels
	vector<vector<float>> sobel_horizontal = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	vector<vector<float>>   sobel_vertical = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	// Separate 2-D kernel to two 1-D kernels for speed boost
	vector<float> sobel_one = { 1, 0, -1 };
	vector<float> sobel_two = { 1, 2, 1 };

	
	Mat magnitude,
		gradient, 
		suppressed;

	int rows, cols, size;









	// Not maintained
	// CannyEdge() use a 3x3 kernel for covlution which is slower than CannyEdge2()
	// void CannyEdge(Mat &src, Mat &dst, float high_thres = 200, float low_thres = 100);






	void cannyEdge_cuda(Mat& src, float& high_thres, float& low_thres);






	// CannEdge2() separate the sobel kernel to two 3-element kernel for convolution,
	// so its faster than CannyEdge().  And the convolution process is further optimized
	// Input param:
	// 		Take a grayscale image as a input
	// Output:
	// 		An edge map result in grayscale
	void cannyEdge2(Mat& src, Mat&dst, float high_thres, float low_thres) {


		this->rows = src.rows;
		this->cols = src.cols;
		this->size = this->rows * this->cols;


		Mat gx(src.rows, src.cols, CV_16SC1); // Short type
		Mat gy(src.rows, src.cols, CV_16SC1); // Short type
		Mat tmp(src.rows, src.cols, CV_16SC1);

		this->conv2_h_sobel<uchar, short>(src, tmp, this->sobel_one);
		this->conv2_v_sobel<short, short>(tmp,  gx, this->sobel_two);
	
		this->conv2_h_sobel<uchar, short>(src, tmp, this->sobel_two);
		this->conv2_v_sobel<short, short>(tmp,  gy, this->sobel_one);


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
		// std::thread mag_thread(&Edge::calculate_Magnitude<short, short>, this, std::ref(gx), std::ref(gy), std::ref(magnitude), true);
		

		// Save gradient result in signed char (schar)
		this->calculate_Gradients<short, short>(gx, gy, this->gradient);
		
		
		this->nonMaxSuppresion(this->magnitude, this->gradient, gy, gx, this->suppressed, high_thres, low_thres);

		this->hysteresis_threshold(dst);

		this->release();

		return;
	} // end of cannyedge2






	void release() 
	{
		magnitude.release();
		gradient.release();
		suppressed.release();
	};




private: 


	// Input 
	//		'Magnitdue'  should be a 8-bit uchar type
	//		'gradient'   should be a 8-bit schar type
	//      'gx' & 'gy'  are two short type convoluted results
	//      'high_thre'  upper threshold value for thresholding
	//      'low_thre'   lower threshold value for thresholding
	// Output:
	// 		'dst'        where to store the suppression result in 8-bit uchar
	void nonMaxSuppresion(
		const Mat& magnitude, 
		const Mat& gradient, 
		const Mat& gy, const Mat& gx, Mat& dst, 
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



		#pragma omp parallel for schedule(dynamic, 1) num_threads(6)
		for (int i = 2; i < this->rows-2; ++i) {
			dst_ptr = dst.ptr<uchar>(i);
			mag_ptr = magnitude.ptr<uchar>(i);
			gra_ptr = gradient.ptr<schar>(i);
			gx_p    = gx.ptr<short>(i);
			gy_p    = gy.ptr<short>(i);


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
											dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : 125;
									}
										
							}
							else if (theta == 0) 
							{
									// horizontal direction
									if (cur_mag_val > mag_ptr[j - 1] && cur_mag_val >= mag_ptr[j + 1]) 
									{
											dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : 125;
									}
										
							}
							else  // bottom-left to top-right  or  bottom-right to top-left direction
							{ 
									int d = (gy_p[j] * gx_p[j] < 0) ? 1 : -1;
									if (cur_mag_val >= mag_ptr[j + cols - d] && cur_mag_val > mag_ptr[j - cols + d]) 
									{
											dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : 125;
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






	void hysteresis_threshold(Mat& src) 
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









	// Input params: 
	//		'src1' & 'src2'    Are gx & gy respectively
	//      'To_8bits'         Turning calculated magnitude result to 8 bits or not
	// Output:
	//		'dst'              Where to store the magnitude result
	template <typename src1_type=short, typename src2_type=short>
	void calculate_Magnitude(const Mat& src1, const Mat& src2, Mat& dst, bool To_8bits = false) 
	{

		if (dst.empty() || dst.type() != CV_32FC1) dst = Mat(src1.rows, src1.cols, CV_32FC1);


		
		#pragma omp parallel for num_threads(8)
		for (int i = 0; i < this->rows; ++i)
		{
			const src1_type* gx = src1.ptr<src1_type>(i);
			const src2_type* gy = src2.ptr<src2_type>(i);
			float* dst_p = dst.ptr<float>(i);

#ifdef __GNUC__
			#pragma omp simd 
#endif		
			for (int j = 0; j < this->cols; ++j) 
			{ 
				dst_p[j] = abs(gy[j]) + abs(gx[j]); // faster & vectorized
				//dst_p[j] = std::sqrt(std::abs(gy[j]*gy[j] + gx[j]*gx[j]));
			}
		}


		if (To_8bits)
			dst.convertTo(dst, CV_8UC1);

#ifdef DEBUG_IMSHOW_RESULT
	imshow("Magnitude result", dst);
	waitKey(10);
#endif 

		return;
	};








	
	// to convert it to degree.
	// Input params: 
	//		'src1' & 'src2'    Are gx & gy respectively
	// Output:
	// 		'dst'              Where to store the gradient result (in degrees) in signed char
	template <typename src1_type, typename src2_type>
	void calculate_Gradients(const Mat& src1, const Mat& src2, Mat& dst) 
	{

		if (dst.empty()) dst = Mat(this->rows, this->cols, CV_8SC1);

		const src1_type* gx;
		const src2_type* gy;
		schar* dst_p;

		float w;

		#pragma omp parallel for num_threads(6)
		for (int i = 0; i < this->rows; ++i)
		{  
			gx = src1.ptr<src1_type>(i);
			gy = src2.ptr<src2_type>(i);
			dst_p = dst.ptr<schar>(i);


	    	for (int j = 0; j < this->cols; ++j)
			{

				w = abs(gy[j] / (gx[j]+0.9));

				if (w < 0.4 || -w > -0.4)
					dst_p[j] = 0;
				else if (w > 2.3 || w < -2.3)
					dst_p[j] = 90;
				else
					dst_p[j] = 45;
// #endif
			}
		}


		return;
	};







	template <typename T>
	bool canny_hysteresis_dfs(T* src_p, const int i, const int j, const int rows, const int cols, const int key)
	{
			int step = i * cols + j;

			// check boundary, make sure instensity isn't equal to zero & key.  key means visted.
			if(i < 0 || j < 0 || i >= rows || j >= cols || src_p[step] == 0 || src_p[step] == key)
					return false;

			// check if found what we want
			else if (*(src_p + i*cols + j) == 255)
					return true;

			// if its a potential edge candidate, keep looking its neighbors
			else if (*(src_p + i*cols + j) == 125)
			{
					*(src_p + i*cols + j) = key; // marked as visited
					
					if( canny_hysteresis_dfs(src_p, i,   j+1, rows, cols, key) ) {	src_p[step] = 255;  return true; }
					if( canny_hysteresis_dfs(src_p, i,   j-1, rows, cols, key) ) {	src_p[step] = 255;  return true; }
					if( canny_hysteresis_dfs(src_p, i+1,   j, rows, cols, key) ) {	src_p[step] = 255;  return true; }
					if( canny_hysteresis_dfs(src_p, i-1,   j, rows, cols, key) ) {	src_p[step] = 255;  return true; }
					if( canny_hysteresis_dfs(src_p, i+1, j+1, rows, cols, key) ) {	src_p[step] = 255;  return true; }
					if( canny_hysteresis_dfs(src_p, i+1, j-1, rows, cols, key) ) {	src_p[step] = 255;  return true; }
					if( canny_hysteresis_dfs(src_p, i-1, j+1, rows, cols, key) ) {	src_p[step] = 255;  return true; }
					if( canny_hysteresis_dfs(src_p, i-1, j-1, rows, cols, key) ) {	src_p[step] = 255;  return true; }
					return false;
			}
			return false;
	};







#ifdef __GNUC__
	#pragma omp declare simd inbranch
#endif
	inline schar simd_w_classifier(float w) { 
		if (w < 0.4)
			return 0;
		else if (w > 2.3)
			return 90;
		else 
			return 45;
	};



}; // end of Edge class

