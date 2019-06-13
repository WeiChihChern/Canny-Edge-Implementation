#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#include "Utils.h"

using namespace std;
using namespace cv;


class Edge : public Utils
{
public:
	vector<vector<float>> sobel_horizontal = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
	vector<vector<float>> sobel_vertical = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

	Edge();
	~Edge();
	Mat getEdge(Mat &src);

private: 
	// square root of the sum of the squares 
	Mat map2edge(const Mat &src1, const Mat &src2);
};

