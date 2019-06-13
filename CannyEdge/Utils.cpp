#include "Utils.h"
#include <algorithm>

//#include "opencv2/opencv.hpp"
//
//using namespace std;

Utils::Utils()
{
}


Utils::~Utils()
{
}


//template <typename T1>
void Utils::conv2(const Mat &src, Mat &dst, const vector< vector<float>> &kernel) {
	if (src.empty()) { cout << "conv2() error. Empty input!\n"; return; }
	if (src.channels() == 3) { cvtColor(src, src, COLOR_BGR2GRAY); }
	// src is now in grayscale
	dst = Mat(src.rows, src.cols, CV_32FC1, cv::Scalar(0));

	int k_rows = kernel.size(),
		k_cols = kernel[0].size(),
		src_rows = src.rows,
		src_cols = src.cols;

	int offset_row = k_rows / 2,
		offset_col = k_cols / 2;
		
	for (int i = offset_row; i < src_rows - offset_row; i++) {
		//uchar* src_ptr = src.ptr<uchar>(i);
		float* dst_ptr = dst.ptr<float>(i);
		
		for (int j = offset_col; j < src_cols - offset_col; j++) {


			float sum = 0; //Kernel could be in float sometimes
			for (int k = -k_rows/2; k < k_rows/2+1; k++) {
				const uchar* src_ptr = src.ptr<uchar>(i+k);
				src_ptr += j;
				for (int l = -k_cols/2; l < k_cols/2+1; l++) {
					sum += (src_ptr[l] * kernel[k+k_rows/2][l+k_cols/2]);
				}
			}
			
			dst_ptr[j] = sum;
		}
	}

	return;
}