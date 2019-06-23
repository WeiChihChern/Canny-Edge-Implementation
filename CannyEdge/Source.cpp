#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

#include "Edge.h"
#include "Timer.h"

using namespace std;
using namespace cv;


int main() {





	Mat small = imread("Capture.PNG", 0);

	Mat big;
	resize(small, big, Size(50000, 5000)); // Benchmark purpose

	GaussianBlur(small, small, Size(3, 3), 0.5);
	GaussianBlur(big, big, Size(3, 3), 0.5);

	Timer timer;
	Edge tool;

	timer.start();
	Mat my_result = tool.cannyEdge2(small);
	timer.stop();

	cout << "My canny edge on small image: " << timer.elapsedMilliseconds() << endl;


	Mat cannyresult;
	timer.start();
	cv::Canny(small, cannyresult, 200, 100, 3,true);
	timer.stop();

	cout << "OpenCV canny edge on small image: " << timer.elapsedMilliseconds() << endl;


	tool.release();
	timer.start();
	my_result = tool.cannyEdge2(big);
	timer.stop();

	cout << "My canny edge on big image: " << timer.elapsedMilliseconds() << endl;


	timer.start();
	cv::Canny(big, cannyresult, 200, 100, 3, true);
	timer.stop();

	cout << "OpenCV canny edge on big image: " << timer.elapsedMilliseconds() << endl;


	return 0;
}