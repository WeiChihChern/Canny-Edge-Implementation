#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

#include "Edge.h"

using namespace std;
using namespace cv;




int main() {



	Mat img = imread("Capture.PNG", 0);
	//resize(img, img, Size(1920,1080));

	GaussianBlur(img, img, Size(3, 3), 0.5);

	Edge tool;
	Mat my_result = tool.cannyEdge2(img);
	imshow("separate", my_result);
	waitKey(10);

	Mat cannyresult;
	cv::Canny(img, cannyresult, 200, 100, 3,true);
	imshow("canny", cannyresult);
	waitKey(0);

	return 0;
}