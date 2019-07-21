#pragma once


#include <vector>


#include "opencv2/opencv.hpp"
#include "Utils.h"


using namespace std;
using namespace cv;


constexpr auto OFFSET   = 0.01;      
typedef  unsigned char  uchar;


#ifdef _DEBUG
	#define DEBUG_IMSHOW_RESULT
#endif






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

	Edge();
	~Edge();








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
	void cannyEdge2(Mat& src, Mat &dst, float high_thres = 200, float low_thres = 100);







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
		float high_thres, float low_thres);




	void hysteresis_threshold(Mat& src);







	// Input params: 
	//		'src1' & 'src2'    Are gx & gy respectively
	//      'To_8bits'         Turning calculated magnitude result to 8 bits or not
	// Output:
	//		'dst'              Where to store the magnitude result
	template <typename src1_type, typename src2_type>
	inline void calculate_Magnitude(const Mat& src1, const Mat& src2, Mat& dst, bool To_8bits = false) 
	{

		if (dst.empty() || dst.type() != CV_32FC1) dst = Mat(src1.rows, src1.cols, CV_32FC1);


		
		#pragma omp parallel for 
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
	inline void calculate_Gradients(const Mat& src1, const Mat& src2, Mat& dst) 
	{

		if (dst.empty()) dst = Mat(this->rows, this->cols, CV_8SC1);

		const src1_type* gx;
		const src2_type* gy;
		schar* dst_p;

		float w;

		#pragma omp parallel for 
		for (int i = 0; i < this->rows; ++i)
		{  
			gx = src1.ptr<src1_type>(i);
			gy = src2.ptr<src2_type>(i);
			dst_p = dst.ptr<schar>(i);

#ifdef __GNUC__
			#pragma omp simd //Vectorized
#endif
	    	for (int j = 0; j < this->cols; ++j)
			{
				w = abs(gy[j] / (gx[j] + OFFSET));

#ifdef __GNUC__
				dst_p[j] = this->simd_w_classifier(w);
#else
				if (w < 0.4)
					dst_p[j] = 0;
				else if (w > 2.3)
					dst_p[j] = 90;
				else
					dst_p[j] = 45;
#endif
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

