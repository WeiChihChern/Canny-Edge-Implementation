#include "Edge.h"

#include <algorithm>
#include <math.h>

Edge::Edge()
{
}


Edge::~Edge()
{
}


Mat Edge::CannyEdge(Mat& src) {
	Mat copy1, copy2;
	this->conv2<uchar>(src, copy1, sobel_horizontal);
	this->conv2<uchar>(src, copy2, sobel_vertical);

	this->calculate_Magnitude(copy1, copy2);
	this->calculate_Gradients(copy1, copy2);
	
	copy1.release();
	copy2.release();

	this->nonMaxSuppresion(magnitude, gradient);

	return suppressed;
}


Mat Edge::cannyEdge2(Mat& src) {


	Mat gx(src.rows, src.cols, CV_32FC1, cv::Scalar(0));
	this->conv2_h<uchar>(       src, gx, this->sobel_one);
	this->conv2_v<float>(gx.clone(), gx, this->sobel_two);


	Mat gy(src.rows, src.cols, CV_32FC1, cv::Scalar(0));
	this->conv2_h<uchar>(       src, gy, this->sobel_two);
	this->conv2_v<float>(gy.clone(), gy, this->sobel_one);

#ifdef DEBUG
	Mat conv2_result;
	copy2.convertTo(conv2_result, CV_8UC1);
	imshow("cannyEdge2's conv2d(), result of conv2_h() & conv2_v() in 8-bit (from float)", conv2_result);
	waitKey(10);
#endif 

	this->calculate_Magnitude(gx, gy);
	this->calculate_Gradients(gx, gy);

	gx.release();
	gy.release();

	this->nonMaxSuppresion(this->magnitude, this->gradient);

	return this->hysteresis_threshold(suppressed);

}


void Edge::calculate_Magnitude(const Mat& src1, const Mat& src2) {
	if(this->magnitude.empty()) this->magnitude = Mat(src1.rows, src1.cols, CV_32FC1);

	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), this->magnitude.begin<float>(), 
		[](const float &s1, const float &s2) 
		{
			return std::sqrt(s1*s1 + s2*s2);
		}
	);

#ifdef DEBUG
	Mat magnitude_show;
	this->magnitude.convertTo(magnitude_show, CV_8UC1);
	imshow("Edge class's calculate_magnitude() result in 8-bit (from float)", magnitude_show);
	waitKey(10);
#endif 

}


void Edge::calculate_Gradients(const Mat& src1, const Mat& src2) {
	
	if(this->gradient.empty()) this->gradient = Mat(src1.rows, src1.cols, CV_32FC1);

	// src2 = G(y) & src1 = G(x)
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), this->gradient.begin<float>(),
		[](const float& s1, const float& s2)
		{
			return std::atan(s2/s1); // G(y) / G(x)
		}
	);

#ifdef DEBUG
	Mat gradient_show;
	this->gradient.convertTo(gradient_show, CV_8UC1);
	imshow("Edge class's calculate_gradient() result in 8-bit (from float)", gradient_show);
	waitKey(10);
#endif 

}


void Edge::nonMaxSuppresion(Mat &magnitude, const Mat &gradient) {
	// Both magnitude & gradient are in float type
	if(this->suppressed.empty()) this->suppressed = Mat (magnitude.rows, magnitude.cols, CV_32FC1, cv::Scalar(0));
	int rows = magnitude.rows,
		cols = magnitude.cols;

	for (int i = 1; i < rows-1; i++) {
			  float* dst_ptr = this->suppressed.ptr<float>(i);
		      float* mag_ptr = magnitude.ptr<float>(i);
		const float* gra_ptr = gradient.ptr<float>(i);
		
		for (int j = 1; j < cols-1; j++) {
			float       angle = gra_ptr[j]*CONSTANT;
			float cur_mag_val = mag_ptr[j];
			angle = (angle < 0) ? 180 + angle : angle;

			if (cur_mag_val != 0) {
				if (angle >= 67.5 && angle < 112.5) {
					// vertical direction
					if (cur_mag_val > mag_ptr[j - cols] && cur_mag_val > mag_ptr[j + cols])
						dst_ptr[j] = cur_mag_val;
				}
				else if ((angle <= 22.5 && angle >= 0) || (angle <= 180 && angle > 157.5)) {
					// horizontal direction
					if (cur_mag_val > mag_ptr[j - 1] && cur_mag_val > mag_ptr[j + 1])
						dst_ptr[j] = cur_mag_val;
				}
				else if ((angle <= 157.5 && angle >= 112.5)) {
					// bottom-left to top-right direction
					if (cur_mag_val > mag_ptr[j + cols - 1] && cur_mag_val > mag_ptr[j - cols + 1])
						dst_ptr[j] = cur_mag_val;
				}
				else if ((angle < 67.5 && angle > 22.5)) {
					// bottom-right to top-left direction
					if (cur_mag_val > mag_ptr[j + cols + 1] && cur_mag_val > mag_ptr[j - cols - 1])
						dst_ptr[j] = cur_mag_val;
				}
				else {
					// assign NaN to zero
					mag_ptr[j] = 0;
				}
			} 
			else {
				dst_ptr[j] = 0;
			}
		}
	}


#ifdef DEBUG
	Mat suppressed_show;
	this->suppressed.convertTo(suppressed_show, CV_8UC1);
	imshow("Edge class's nonMaxSuppression() result in 8-bit (from float)", suppressed_show);
	waitKey(10);
#endif 

}


Mat Edge::hysteresis_threshold(Mat& src, float high_thres, float low_thres) {

	if (src.empty() || src.channels() == 3) { cout << "hysteresis_threshold() error!\n"; return Mat(0,0,CV_8UC1); }


	// Using two threshold values to perform hysteresis thresholding
	std::transform(src.begin<float>(), src.end<float>(), src.begin<float>(), 
		[&high_thres, &low_thres] (const float &src_val) {
			if (src_val >= high_thres) 
				return 255;
			else if (src_val < high_thres && src_val >= low_thres)
				return 125;
			else 
				return 0;
		});
	src.convertTo(src, CV_8UC1);



	// After thresholding, if pixel is assigned as 125,
	// check that pixel's 8-neighbor to see if any strong
	// pixel with intensity value of 255 exists. If so,
	// that pixel can be assigned as a strong pixel with
	// 255 intensity value. Otherwise, suppress it to 0
	Mat dst(src.rows, src.cols, CV_8UC1, Scalar(0));
	int cols = src.cols;
	for (int i = 1; i < src.rows-1; i++) {
		uchar* dst_ptr = dst.ptr<uchar>(i);
		uchar* src_ptr = src.ptr<uchar>(i);
			
		for (int j = 1; j < cols-1; j++) {
			if (src_ptr[j] == 125) { // potential strong edge pixel
				if (src_ptr[j - 1] == 255) dst_ptr[j] = 255;
				else if (src_ptr[j + 1]        == 255) dst_ptr[j] = 255;
				else if (src_ptr[j - cols]     == 255) dst_ptr[j] = 255;
				else if (src_ptr[j + cols]     == 255) dst_ptr[j] = 255;
				else if (src_ptr[j - cols - 1] == 255) dst_ptr[j] = 255;
				else if (src_ptr[j - cols + 1] == 255) dst_ptr[j] = 255;
				else if (src_ptr[j + cols + 1] == 255) dst_ptr[j] = 255;
				else if (src_ptr[j + cols - 1] == 255) dst_ptr[j] = 255;
			}
			else if (src_ptr[j] == 255) dst_ptr[j] = 255;
		}
	}



#ifdef DEBUG
	Mat threshold_show;
	dst.convertTo(threshold_show, CV_8UC1);
	imshow("Edge class's hysteresis_threshold() result in 8-bit (from float)", threshold_show);
	waitKey(10);
#endif 

	return dst;
}