#pragma once

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


// This class is for some general utilities' functions

class Utils
{

public:
	Utils();
	~Utils();
	

	void conv2(const Mat& src, Mat& dst, const vector< vector<float>> &kernel);
	


	/* 
		Added for separable kernel to enhance conv2's speed
		conv2_v() is for 1-D kernel like 3x1, 5x1, Nx1 kernel
		conv2_h() is for 1-D kernel like 1x3, 1x5, 1xN kernel
	*/
	//template<typename T1>
	void conv2_v(const Mat& src, Mat& dst, const vector<float> kernel);
	void conv2_h(const Mat& src, Mat& dst, const vector<float> kernel);


};

