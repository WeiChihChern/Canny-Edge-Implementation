#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

#include "Edge.h"
#include "Timer.h"

using namespace std;
using namespace cv;



int main(int argc, char* argv[]) {

	Timer timer;
	Edge tool;
	int iterations = 1000;
	Mat result, 
		big, 
		input, 
		small;


	bool _big = false;
	if (strcmp(argv[1], "-large") == 0)
	{
		_big = true;
		cout << "Using Large image (4K) for benchmark!\n";
	}
	else
	{
		cout << "Using small image (637 x 371) for benchmark!\n";
	}

	if (strcmp(argv[2], "-iter") == 0)
	{
		iterations = atoi(argv[3]);
		cout << "Iteration set to: " << iterations << "\n";
	}



	small = imread("Capture.PNG", 0);

	if (_big)
	{
		resize(small, big, Size(3840, 2160)); // For benchmark only
		input = big;
	}
	else
	{
		GaussianBlur(small, small, Size(3, 3), 0.5);
		input = small;
	}


	
	
	

	timer.start();
	for (int i = 0; i < iterations; i++) {
		tool.cannyEdge2(input, result, 200, 100);
	}
	timer.stop(); 
	cout << "My canny edge on small image: " << timer.elapsedMilliseconds() / (double)iterations << "ms (avg of " 
		<< to_string(iterations) << " runs)\n";

	//imshow("My canny edge", result);
	//waitKey(10);



	timer.start();
	for (int i = 0; i < iterations; i++) {
		cv::Canny(input, result, 200, 100);
	}
	timer.stop();

	cout << "Opencv canny edge on small image: " << timer.elapsedMilliseconds() / (double)iterations 
		<< "ms (avg of " << to_string(iterations) << " runs)\n";


	//imshow("Opencv canny edge", result);
	//waitKey(10);


	return 0;
}