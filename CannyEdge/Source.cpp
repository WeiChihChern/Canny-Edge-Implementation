#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

#include "Edge.h"

using namespace std;
using namespace cv;


template <typename T1>
void conv2_v(const Mat& src, Mat& dst, const vector<T1> kernel) {

	int k_size = kernel.size(),
		src_rows = src.rows,
		src_cols = src.cols;

	int offset_row = k_size / 2;
	int k_idx = -offset_row;

	for (int i = offset_row; i < (src.rows - offset_row); i++) {
		float* dst_ptr = dst.ptr<float>(i);

		for (int j = 0; j < src_cols; j++) {
			const float* src_ptr = src.ptr<float>(i) + j;
			float sum = 0;

			//std::for_each(kernel.begin(), kernel.end(),
			//	[src_ptr, &sum, k_idx, &src_cols] (const float& k_val) mutable
			//	{
			//		const float *val = src_ptr - (k_idx++ * src_cols);
			//		sum += (*val * k_val);
			//	}
			//);

			// For loop proven to be faster than std::for_each in this case
			for (int k = 0; k < kernel.size(); k++)
			{
				int pixel_idx = (k - offset_row) * src_cols;
				sum += src_ptr[pixel_idx] * kernel[k];
			}


			dst_ptr[j] = sum;
		}
	}
}

int main() {



	Mat img = imread("Capture.PNG", 0);

	Mat dst(img.rows, img.cols, CV_32FC1);

	vector<int> kernel = { -1,0,1 };

	conv2_v(img, dst, kernel);

	GaussianBlur(img, img, Size(3, 3), 0.5);

	Edge tool;
	Mat img2 = tool.cannyEdge2(img);

	Mat img3; 
	img2.convertTo(img3, CV_8UC1); // Edge result is in float type, convert it to 8-bit for display purpose only
	imshow("separate", img3);
	waitKey(0);



	Mat normal = tool.CannyEdge(img);
	Mat new_normal;
	normal.convertTo(new_normal, CV_8UC1);
	imshow("normal", new_normal);
	waitKey(10);

	return 0;
}