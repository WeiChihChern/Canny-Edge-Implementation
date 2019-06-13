#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

#include "Edge.h"

using namespace std;
using namespace cv;


int main() {


	Mat img = imread("Capture.PNG", 0);


	Edge edge;
	Mat img2 = edge.getEdge2(img);

	Mat img3; 
	img2.convertTo(img3, CV_8UC1); // Edge result is in float type, convert it to 8-bit for display purpose only
	imshow("separate", img3);
	waitKey(10);

	Mat normal = edge.getEdge(img);
	Mat new_normal;
	normal.convertTo(new_normal, CV_8UC1);
	imshow("normal", new_normal);
	waitKey(10);

	return 0;
}