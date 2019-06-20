#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include "Utils.h"

using namespace std;
using namespace cv;

constexpr auto PI = 3.14159265;
constexpr auto TO_THETA = 180 / PI;
constexpr auto OFFSET   = 0.01;


#if 1
	#define USE_SIMPLE_LOOP
#else
	#define DEBUG_IMSHOW_RESULT
	// #define SHOW_GRADIENT_RESULT
	// #define NonMaxSuppress_SHOW_THETA
#endif


class Edge : public Utils
{
public:
	vector<vector<float>> sobel_horizontal = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	vector<vector<float>>   sobel_vertical = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };
	vector<float>                sobel_one = { 1, 0, -1 };
	vector<float>                sobel_two = { 1, 2, 1 };
	Mat magnitude, gradient, suppressed;

	Edge();
	~Edge();
	Mat CannyEdge(Mat &src, float high_thres = 200, float low_thres = 100);
	Mat cannyEdge2(Mat& src, float high_thres = 200, float low_thres = 100);
private: 


	// Square root of the sum of the squares -> ( G(x)^2 + G(y)^2 )^0.5
	// Return a CV_32FC1 type magnitude map
	inline void calculate_Magnitude(const Mat &src1, const Mat &src2);
	
	// Using L2 norm gradient, which uses atan() for gradient calculation.
	// The most expensive part of canny edge detection.
	// Return a CV_32FC1 type gradient map
	inline void calculate_Gradients(const Mat& src1, const Mat& src2);
	
	// Input: 
	//	Magnitdue & Gradient are both in CV_32FC1 type
	// Return a suppressed result in CV_32FC1
	void nonMaxSuppresion(Mat& magnitude, const Mat& gradient);


	// src is the result of non maximum suppression (CV_)
	// Mat will be converted to CV_8UC1
	Mat hysteresis_threshold(Mat& src, float high_thres = 200, float low_thres = 100);
	
};

