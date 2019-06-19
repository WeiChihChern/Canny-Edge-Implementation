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
	this->conv2(src, copy1, sobel_horizontal);
	this->conv2(src, copy2, sobel_vertical);

	this->magnitude = this->calculate_Magnitude(copy1, copy2);
	this->gradient  = this->calculate_Gradients(copy1, copy2);
	
	copy1.release();
	copy2.release();

	this->suppressed = this->nonMaxSuppresion(magnitude, gradient);

	return suppressed;
}


Mat Edge::cannyEdge2(Mat& src) {

	// copy1 = G(x)
	Mat copy1(src.rows, src.cols, CV_32FC1, cv::Scalar(0));
	this->conv2_h(          src, copy1, this->sobel_one);
	this->conv2_v(copy1.clone(), copy1, this->sobel_two);

	// copy2 = G(y)
	Mat copy2(src.rows, src.cols, CV_32FC1, cv::Scalar(0));
	this->conv2_h(          src, copy2, this->sobel_two);
	this->conv2_v(copy2.clone(), copy2, this->sobel_one);

#ifdef DEBUG
	Mat conv2_result;
	copy2.convertTo(conv2_result, CV_8UC1);
	imshow("cannyEdge2's conv2d(), result of conv2_h() & conv2_v() in 8-bit (from float)", conv2_result);
	waitKey(10);
#endif 

	this->magnitude = this->calculate_Magnitude(copy1, copy2);

	this->gradient = this->calculate_Gradients(copy1, copy2);

	this->suppressed = this->nonMaxSuppresion(this->magnitude, this->gradient);

	this->hysteresis_threshold(suppressed);


	return suppressed;
}

Mat Edge::calculate_Magnitude(const Mat& src1, const Mat& src2) {
	Mat result(src1.rows, src1.cols, CV_32FC1);
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), result.begin<float>(), 
		[](const float &s1, const float &s2) 
		{
			return std::sqrt(s1*s1 + s2*s2);
		}
	);

#ifdef DEBUG
	Mat magnitude_show;
	result.convertTo(magnitude_show, CV_8UC1);
	imshow("Edge class's calculate_magnitude() result in 8-bit (from float)", magnitude_show);
	waitKey(10);
#endif 


	return result;
}


Mat Edge::calculate_Gradients(const Mat& src1, const Mat& src2) {
	
	Mat result(src1.rows, src1.cols, CV_32FC1);
	// src2 = G(y) & src1 = G(x)
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), result.begin<float>(),
		[](const float& s1, const float& s2)
		{
			return std::round(std::atan(s2/s1) * CONSTANT); // G(y) / G(x)
		}
	);

#ifdef DEBUG
	Mat gradient_show;
	result.convertTo(gradient_show, CV_8UC1);
	imshow("Edge class's calculate_gradient() result in 8-bit (from float)", gradient_show);
	waitKey(10);
#endif 

	return result;
}


Mat Edge::nonMaxSuppresion(Mat &magnitude, const Mat &gradient) {
	// Both magnitude & gradient are in float type
	Mat result(magnitude.rows, magnitude.cols, CV_32FC1, cv::Scalar(0));
	int rows = magnitude.rows,
		cols = magnitude.cols;

	for (int i = 1; i < rows-1; i++) {
			  float* dst_ptr = result.ptr<float>(i);
		      float* mag_ptr = magnitude.ptr<float>(i);
		const float* gra_ptr = gradient.ptr<float>(i);
		
		for (int j = 1; j < cols-1; j++) {
			float         angle = gra_ptr[j];
			float cur_pixel_val = mag_ptr[j];
			angle = (angle < 0) ? (180 + angle) : angle;

			if (angle >= 67.5 && angle < 112.5) {
				// vertical direction
				if (cur_pixel_val > mag_ptr[j - cols] && cur_pixel_val > mag_ptr[j + cols])
					dst_ptr[j] = cur_pixel_val;
			}
			else if ((angle <= 22.5 && angle >= 0) || (angle <= 180 && angle > 157.5)) {
				// horizontal direction
				if (cur_pixel_val > mag_ptr[j - 1] && cur_pixel_val > mag_ptr[j + 1])
					dst_ptr[j] = cur_pixel_val;
			}
			else if ((angle <= 157.5 && angle >= 112.5)) {
				// bottom-left to top-right direction
				if (cur_pixel_val > mag_ptr[j + cols - 1] && cur_pixel_val > mag_ptr[j - cols + 1])
					dst_ptr[j] = cur_pixel_val;
			}
			else if ((angle < 67.5 && angle > 22.5)) {
				// bottom-right to top-left direction
				if (cur_pixel_val > mag_ptr[j + cols + 1] && cur_pixel_val > mag_ptr[j - cols - 1])
					dst_ptr[j] = cur_pixel_val;
			}
			else { 
				// assign NaN to zero
				mag_ptr[j] = 0;
			}
		}
	}


#ifdef DEBUG
	Mat suppressed_show;
	result.convertTo(suppressed_show, CV_8UC1);
	imshow("Edge class's nonMaxSuppression() result in 8-bit (from float)", suppressed_show);
	waitKey(10);
#endif 


	return result;
}


void Edge::hysteresis_threshold(Mat& src, float high_thres, float low_thres) {
	if (src.empty() || src.channels() == 3) { cout << "hysteresis_threshold() error!\n"; return; }

	std::transform(src.begin<float>(), src.end<float>(), src.begin<float>(), [&high_thres, &low_thres] (float &val) 
		{
			if (val >= high_thres) 
				return 255;
			else if (val < high_thres && val >= low_thres) 
				return 125;
			else 
				return 0;
		});


#ifdef DEBUG
	Mat threshold_show;
	src.convertTo(threshold_show, CV_8UC1);
	imshow("Edge class's hysteresis_threshold() result in 8-bit (from float)", threshold_show);
	waitKey(10);
#endif 

	return;
}