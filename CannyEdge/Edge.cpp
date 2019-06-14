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
	Mat copy1(src.rows, src.cols, CV_32FC1, cv::Scalar(0));
	this->conv2_h(          src, copy1, this->sobel_one);
	this->conv2_v(copy1.clone(), copy1, this->sobel_two);
	
	//copy1.convertTo(copy1, CV_8UC1);
	//imshow("test", copy1);
	//waitKey(0);

	Mat copy2(src.rows, src.cols, CV_32FC1, cv::Scalar(0));
	this->conv2_h(          src, copy2, this->sobel_two);
	this->conv2_v(copy2.clone(), copy2, this->sobel_one);

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
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), result.begin<float>(),
		[](const float& s1, const float& s2)
		{
			return std::atan(s1 * s1 + s2 * s2);
		}
	);
	return result;
}