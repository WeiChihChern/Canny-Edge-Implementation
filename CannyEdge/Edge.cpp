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