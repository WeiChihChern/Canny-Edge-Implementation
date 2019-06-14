#include "Edge.h"

#include <algorithm>
#include <math.h>

Edge::Edge()
{
}


Edge::~Edge()
{
}


Mat Edge::getEdge(Mat& src) {
	Mat copy1, copy2;
	this->conv2(src, copy1, sobel_horizontal);
	this->conv2(src, copy2, sobel_vertical);

	return this->getMagnitude(copy1, copy2);
};


Mat Edge::getEdge2(Mat& src) {

	// copy1 = G(x)
	Mat copy1(src.rows, src.cols, CV_32FC1, cv::Scalar(0));
	this->conv2_h(          src, copy1, this->sobel_one);
	this->conv2_v(copy1.clone(), copy1, this->sobel_two);
	
#ifdef DEBUG
	Mat getEdge2;
	copy1.convertTo(getEdge2, CV_8UC1);
	imshow("getEdge2(), result of conv2_h() & conv2_v() in 8-bit", getEdge2);
	waitKey(0);
#endif 

	// copy2 = G(y)
	Mat copy2(src.rows, src.cols, CV_32FC1, cv::Scalar(0));
	this->conv2_h(          src, copy2, this->sobel_two);
	this->conv2_v(copy2.clone(), copy2, this->sobel_one);

	this->getGradients(copy1, copy2);
	return this->getMagnitude(copy1, copy2);
}

Mat Edge::getMagnitude(const Mat& src1, const Mat& src2) {
	Mat result(src1.rows, src1.cols, CV_32FC1);
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), result.begin<float>(), 
		[](const float &s1, const float &s2) 
		{
			return std::sqrt(s1*s1 + s2*s2);
		}
	);
	return result;
}


Mat Edge::getGradients(const Mat& src1, const Mat& src2) {
	
	Mat result(src1.rows, src1.cols, CV_32FC1);
	// src2 = G(y) & src1 = G(x)
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), result.begin<float>(),
		[](const float& s1, const float& s2)
		{
			return std::atan(s2/s1) * CONSTANT; // G(y) / G(x)
		}
	);

#ifdef DEBUG
	Mat getGradients;
	result.convertTo(getGradients, CV_8UC1);
	imshow("getGradients() testing in 8-bit (from float type)", getGradients);
	waitKey(0);
#endif 

	return result;
}


Mat Edge::nonMaxSuppresion(Mat &magnitude, const Mat &gradient) {
	// Both magnitude & gradient are in float type

	int rows = magnitude.rows,
		cols = magnitude.cols;

	for (int i = 1; i < rows-1; i++) {
		      float* mag_ptr = magnitude.ptr<float>(i);
		const float* gra_ptr =  gradient.ptr<float>(i);
		
		for (int j = 1; j < cols-1; j++) {
			const float angle   = gra_ptr[j];
			      float cur_pixel_val = mag_ptr[j];

			if (angle >= 67.5 && angle < 112.5) {
				// vertical direction
				mag_ptr[j] = (direction)
			}
			else if ((angle <= 22.5 && angle >= 0) || (angle <= 180 && angle > 157.5)) {
				// horizontal direction
			}
			else if ((angle <= 157.5 && angle >= 112.5)) {
				// bottom-left to top-right direction
			}
			else if ((angle < 67.5 && angle > 22.5)) {
				// bottom-right to top-left direction
			}
			else
		}
	}
}