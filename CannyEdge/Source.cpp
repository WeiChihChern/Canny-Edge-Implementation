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
	resize(small, big, Size(3840, 2160)); // For benchmark only

	GaussianBlur(small, small, Size(3, 3), 0.5);

	Timer timer;
	Edge tool;
	int iterations = 1;
	Mat result, 
		input = big;
	


	timer.start();
	for (int i = 0; i < iterations; i++) {
		tool.cannyEdge2(input, result, 200, 100);
	}
	timer.stop();
	cout << "My canny edge on small image: " << timer.elapsedMilliseconds() / (double)iterations << "ms\n";

	imshow("My canny edge", result);
	waitKey(10);



	timer.start();
	for (int i = 0; i < iterations; i++) {
		cv::Canny(input, result, 200, 100);
	}
	timer.stop();

	cout << "Opencv canny edge on small image: " << timer.elapsedMilliseconds() / (double)iterations << "ms\n";


	imshow("Opencv canny edge", result);
	waitKey(10);


	return 0;
}