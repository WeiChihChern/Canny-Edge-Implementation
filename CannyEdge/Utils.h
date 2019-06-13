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
	
	//template <typename T1>
	void conv2(const Mat& src, Mat& dst, const vector< vector<float>> &kernel);


	void conv2_v(const Mat& src, Mat& dst, const vector<float> kernel);
	void conv2_h(const Mat& src, Mat& dst, const vector<float> kernel);


};

