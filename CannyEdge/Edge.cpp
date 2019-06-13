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

	return this->map2edge(copy1, copy2);
};


Mat Edge::getEdge2(Mat& src) {
	Mat copy1(src.rows, src.cols, CV_32FC1);
	this->conv2_h(src, copy1, this->sobel_one);
	Mat copy1_temp = copy1.clone();
	this->conv2_v(copy1_temp, copy1, this->sobel_two);

	Mat copy2(src.rows, src.cols, CV_32FC1);
	this->conv2_h(src, copy2, this->sobel_two);
	Mat copy2_temp = copy2.clone();
	this->conv2_v(copy2_temp, copy2, this->sobel_one);

	return this->map2edge(copy1, copy2);
}

Mat Edge::map2edge(const Mat& src1, const Mat& src2) {
	Mat result(src1.rows, src1.cols, CV_32FC1);
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), result.begin<float>(), 
		[](const float &s1, const float &s2) 
		{
			return std::sqrt(s1*s1+s2*s2);
		}
	);
	return result;
}


