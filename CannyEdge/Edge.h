#pragma once
#include <vector>


#include "opencv2/opencv.hpp"
#include "Utils.h"


using namespace std;
using namespace cv;


constexpr auto PI = 3.14159265;
constexpr auto TO_THETA = 180 / PI;  // Turn atan(Gy/Gx) to theta
//constexpr auto OFFSET   = 0.01;      



#if 1 // Relase
	/*  for-loop is faster (tested on VS Studio 2019/2015 with OpenCV 4.0.1)
		Disable this will use std::transform + lambda for looping in stead  */
	#define USE_SIMPLE_LOOP 


	#ifdef _OPENMP
		#include <omp.h>

		#ifdef __GNUC__
			#define OMP_FOR(n)  _Pragma("omp parallel for if (n>300000)")
		#elif _MSC_VER
			#define OMP_FOR(n)  __pragma(omp parallel for if (n>100)) // Ability to toggle openmp according to size
		#endif	
	#else
		#define omp_get_thread_num() 0
		#define OMP_FOR(n)
	#endif // _OPENMP



#else // Debug

	#define USE_SIMPLE_LOOP 


	/*  Enable this will imshow conv2D, manitude, gradient, nonMax & thresholding
		result in 8-bit  */
	 #define DEBUG_IMSHOW_RESULT

	// #define DEBUG_SHOW_GRADIENT_RESULT

	 #define DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS

	// #define DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
#endif






class Edge : public Utils // Utils includes 2-D & 1-D convolution functions
{
public:
	// Two sobel 2-D kernels
	vector<vector<float>> sobel_horizontal = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	vector<vector<float>>   sobel_vertical = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	// Separate 2-D kernel to two 1-D kernels for speed boost
	vector<float>                sobel_one = { 1, 0, -1 };
	vector<float>                sobel_two = { 1, 2, 1 };

	
	Mat magnitude, 
		gradient, 
		suppressed;


	Edge();
	~Edge();








	
	// CannyEdge() use a 3x3 kernel for covlution which is slower than CannyEdge2()
	void CannyEdge(Mat &src, Mat &dst, float high_thres = 200, float low_thres = 100);









	// CannEdge2() separate the sobel kernel to two 3-element kernel for convolution,
	// so its faster than CannyEdge().  And the convolution process is further optimized
	// to avoid an extra for-loop
	// Input param:
	//		Input 'src' should be a 8-bit (uchar) grayscale image
	// Output param:
	//		Function will output a 8-bit uchar grayscale image with edges
	void cannyEdge2(Mat& src, Mat &dst, float high_thres = 200, float low_thres = 100);









	void release() {
		magnitude.release();
		gradient.release();
		suppressed.release();
	};




private: 




	// This function uses square root of the sum of the squares: ( G(x)^2 + G(y)^2 )^0.5
	// Input params: 
	//		User can manually select what type your inputs are ('src1' & 'src2')
	// Output:
	//		Will save a uchar result to member variable 'magnitude'
	template <typename src1_type, typename src2_type>
	inline void calculate_Magnitude(const Mat& src1, const Mat& src2, bool To_8bits = false) {

		if (this->magnitude.empty() || this->magnitude.type() != CV_32FC1) this->magnitude = Mat(src1.rows, src1.cols, CV_32FC1);

#ifndef USE_SIMPLE_LOOP
		std::transform(src1.begin<src1_type>(), src1.end<src1_type>(), src2.begin<src2_type>(), this->magnitude.begin<float>(),
			[](const src1_type& s1, const src2_type& s2)
			{
				return std::sqrt(s1 * s1 + s2 * s2);
			}
		);
#else
		int rows = src1.rows,
			cols = src1.cols;

		OMP_FOR(rows*cols) // Automatically ignored if no openmp support
		for (int i = 0; i < src1.rows; i++) {
			const src1_type* gx = src1.ptr<src1_type>(i);
			const src2_type* gy = src2.ptr<src2_type>(i);
			float* dst = this->magnitude.ptr<float>(i);

			for (int j = 0; j < src1.cols; j++) {
				dst[j] = std::sqrt(gy[j] * gy[j] + gx[j] * gx[j]);
			}
		}
#endif


		if (To_8bits)
			this->magnitude.convertTo(this->magnitude, CV_8UC1);


#ifdef DEBUG_IMSHOW_RESULT
		if (this->magnitude.depth() != CV_8UC1) {
			Mat magnitude_show;
			this->magnitude.convertTo(magnitude_show, CV_8UC1);
			imshow("calculate_magnitude() result in 8-bit (from float)", magnitude_show);
		}
		else {
			imshow("calculate_magnitude() result in 8-bit (from float)", this->magnitude);
		}
		waitKey(10);

#endif 

		return;
	};









	// This function uses std::atan() to calculate gradient and multiply a constexpr 'TO_THETA'
	// to convert it to degree.
	// Input params: 
	//		User can manually select what type your inputs are ('src1' & 'src2')
	// Output:
	//		Will save a uchar result to member variable 'gradient'
	template <typename src1_type, typename src2_type>
	inline void calculate_Gradients(const Mat& src1, const Mat& src2) {

		// Result theta range will be within -90 ~ 90, using signed char to store 
		if (this->gradient.empty()) this->gradient = Mat(src1.rows, src1.cols, CV_8SC1);

		int rows = src1.rows,
			cols = src1.cols;

#ifndef USE_SIMPLE_LOOP
		// src2 = G(y) & src1 = G(x)
		std::transform(src1.begin<src1_type>(), src1.end<src1_type>(), src2.begin<src2_type>(), this->gradient.begin<schar>(),
			[](const src1_type& gx, const src2_type& gy)
			{
				if (gx[j] == 0 && gy[j] != 0)
					return (schar)90;
				else if (gy[j] == 0)
					return (schar)0;
				else if (gy[j] / gx[j] == 1)
					return (schar)45;
				else if (gy[j] / gx[j] == -1)
					return (schar)-45;
				else {
					return (schar)(std::atan((float)gy[j] / (float)gx[j]) * TO_THETA);
				}
				);

#else
		OMP_FOR(rows * cols) // Automatically ignored if no openmp support
		for (int i = 0; i < rows; i++) {  // Looping is faster than std::transform on VS 2019
			const src1_type*  gx = src1.ptr<src1_type>(i);
			const src2_type*  gy = src2.ptr<src2_type>(i);
			          schar* dst = this->gradient.ptr<schar>(i);

			// Two if statement to improve speed, atan() is expensive
			for (int j = 0; j < cols; j++) {
				if (gx[j] == 0 && gy[j] != 0)
					dst[j] = (schar)90;
				else if (gy[j] == 0)
					dst[j] = (schar)0;
				else if (gy[j] / gx[j] == 1)
					dst[j] = (schar)45;
				else if (gy[j] / gx[j] == -1)
					dst[j] = (schar)-45;
				else {
					dst[j] = (schar)(std::atan((float)gy[j] / (float)gx[j]) * TO_THETA);
#ifdef DEBUG_SHOW_GRADIENT_RESULT
					cout << (int)dst[j] << " : y=" << gy[j] << ", x=" << gx[j] << endl;
#endif
				}
			}
		}


#endif // USE_SIMPLE_LOOPDEBUG


#ifdef DEBUG_IMSHOW_RESULT // No much info to visualize gradient
		/*Mat gradient_show;
		this->gradient.convertTo(gradient_show, CV_8UC1);
		imshow("calculate_gradient() result in 8-bit (from float)", gradient_show);
		waitKey(10);*/
#endif 

		return;
	};











	// Input params: 
	//		Magnitdue should be in 8-bit uchar type
	//		gradient should be in 8-bit schar type, storing -90 ~ 90 degrees
	// Output:
	//		Will save a uchar result to member variable 'suppressed'
	void nonMaxSuppresion(Mat& magnitude, const Mat& gradient);












	// Input params: 
	//		'src' should be in 8-bit uchar type
	// Output:
	//		will do thresholding inplace in member variable 'suppressed'
	Mat hysteresis_threshold(Mat& src, float high_thres = 200, float low_thres = 100);
	
};

