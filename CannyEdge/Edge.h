#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include "Utils.h"

using namespace std;
using namespace cv;

constexpr auto PI = 3.14159265;
constexpr auto CONSTANT = 180 / PI;
constexpr float highThreshold = 200;
constexpr float lowThreshold  = 100;

#if 0

#else
	#define DEBUG
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
	Mat CannyEdge(Mat &src);
	Mat cannyEdge2(Mat& src);
private: 
	// square root of the sum of the squares 
	Mat calculate_Magnitude(const Mat &src1, const Mat &src2);
	
	Mat calculate_Gradients(const Mat& src1, const Mat& src2);
	
	Mat nonMaxSuppresion(Mat& magnitude, const Mat& gradient);

	void hysteresis_threshold(Mat& src, float high_thres = 200, float low_thres = 100);
	
};

