#pragma once
#include <vector>

#include "opencv2/opencv.hpp"
#include "Utils.h"

using namespace std;
using namespace cv;

constexpr auto PI = 3.14159265;
constexpr auto TO_THETA = 180 / PI;  // Turn atan(Gy/Gx) to theta
//constexpr auto OFFSET   = 0.01;      


#if 0
	// for-loop is faster (tested on VS Studio 2019 with OpenCV 4.0.1)
	// Disable this will use std::transform + lambda for looping instead
	#define USE_SIMPLE_LOOP 

	//#define DEBUG_SHOW_GRADIENT_RESULT
	//#define DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS
	//#define DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
#else
	// Enable this will imshow conv2D, manitude, gradient, nonMax & thresholding 
    // result in 8-bit
	#define DEBUG_IMSHOW_RESULT
	#define USE_SIMPLE_LOOP 
	// #define DEBUG_SHOW_GRADIENT_RESULT
	// #define DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS
	// #define DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
#endif


class Edge : public Utils
{
public:
	vector<vector<float>> sobel_horizontal = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	vector<vector<float>>   sobel_vertical = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	vector<float>                sobel_one = { 1, 0, -1 };
	vector<float>                sobel_two = { 1, 2, 1 };

	Mat magnitude, 
		gradient, 
		suppressed;


	Edge();
	~Edge();
	
	// CannyEdge() use a 3x3 kernel for covlution which is slower than CannyEdge2()
	Mat CannyEdge(Mat &src, float high_thres = 200, float low_thres = 100);

	// CannEdge2() separate the sobel kernel to two 3-element kernel for convolution,
	// so its faster than CannyEdge().  And the convolution process is further optimized
	// to avoid an extra for-loop
	Mat cannyEdge2(Mat& src, float high_thres = 200, float low_thres = 100);

private: 


	// Square root of the sum of the squares -> ( G(x)^2 + G(y)^2 )^0.5
	// Two inputs, src1 & src2 are two short-type sobel filtered result.
	// Output calculation into float-type Mat
	template <typename src1_type, typename src2_type>
	inline void calculate_Magnitude(const Mat& src1, const Mat& src2, bool To_8bits = false) {
		if (this->magnitude.empty()) this->magnitude = Mat(src1.rows, src1.cols, CV_32FC1);

#ifndef USE_SIMPLE_LOOP
		std::transform(src1.begin<src1_type>(), src1.end<src1_type>(), src2.begin<src2_type>(), this->magnitude.begin<float>(),
			[](const src1_type& s1, const src2_type& s2)
			{
				return std::sqrt(s1 * s1 + s2 * s2);
			}
		);
#else
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



	
	// Using L2 norm gradient, which uses atan() for gradient calculation.
	// The most expensive part of canny edge detection.
	// Return a CV_32FC1 type gradient map
	template <typename src1_type, typename src2_type>
	inline void calculate_Gradients(const Mat& src1, const Mat& src2) {

		// Result theta range will be within -90 ~ 90, using signed char to store 
		if (this->gradient.empty()) this->gradient = Mat(src1.rows, src1.cols, CV_8SC1);

#ifndef USE_SIMPLE_LOOP
		// src2 = G(y) & src1 = G(x)
		std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), this->gradient.begin<float>(),
			[](const float& gx, const float& gy)
			{
				if (gx[j] == 0 && gy[j] != 0)
					return 90;
				else if (gx[j] == 0 && gy[j] == 0)
					return 0;
				else {
					return (std::atan(gy[j] / gx[j]) * TO_THETA);
				}
				);

#else

		// Looping is faster than std::transform
		for (int i = 0; i < src1.rows; i++) {
			const src1_type*  gx = src1.ptr<src1_type>(i);
			const src2_type*  gy = src2.ptr<src2_type>(i);
			          schar* dst = this->gradient.ptr<schar>(i);

			// Two if statement to improve speed, atan() is expensive
			for (int j = 0; j < src1.cols; j++) {
				if (gx[j] == 0 && gy[j] != 0)
					dst[j] = 90;
				else if (gx[j] == 0 && gy[j] == 0)
					dst[j] = 0;
				else {
#ifdef DEBUG_SHOW_GRADIENT_RESULT
					cout << std::atan(gy[j] / gx[j]) << " : y=" << gy[j] << ", x=" << gx[j] << endl;
#endif
					dst[j] = std::atan(gy[j] / gx[j]) * TO_THETA;
				}
			}
		}

		double mi, max;
		minMaxLoc(this->gradient, &mi, &max);

#endif // USE_SIMPLE_LOOPDEBUG


#ifdef DEBUG_IMSHOW_RESULT
		Mat gradient_show;
		this->gradient.convertTo(gradient_show, CV_8UC1);
		imshow("calculate_gradient() result in 8-bit (from float)", gradient_show);
		waitKey(10);
#endif 

		return;
	};
	



	// Input: 
	//	Magnitdue & Gradient are both in CV_32FC1 type
	// Return a suppressed result in CV_32FC1
	void nonMaxSuppresion(Mat& magnitude, const Mat& gradient);


	// src is the result of non maximum suppression (CV_)
	// Mat will be converted to CV_8UC1
	Mat hysteresis_threshold(Mat& src, float high_thres = 200, float low_thres = 100);
	
};

