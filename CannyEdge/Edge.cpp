#include "Edge.h"

#include <algorithm>
#include <math.h>

Edge::Edge()
{
}


Edge::~Edge()
{
}


Mat Edge::CannyEdge(Mat& src, float high_thres, float low_thres) {
	Mat copy1, copy2;
	this->conv2<uchar>(src, copy1, sobel_horizontal);
	this->conv2<uchar>(src, copy2, sobel_vertical);

	this->calculate_Magnitude(copy1, copy2);
	this->calculate_Gradients(copy1, copy2);
	
	copy1.release();
	copy2.release();

	this->nonMaxSuppresion(magnitude, gradient);

	return this->hysteresis_threshold(this->suppressed, high_thres, low_thres);
}


Mat Edge::cannyEdge2(Mat& src, float high_thres, float low_thres) {


	Mat gx(src.rows, src.cols, CV_32FC1);
	this->conv2_h_sobel<uchar>(       src, gx, this->sobel_one);
	this->conv2_v_sobel<float>(gx.clone(), gx, this->sobel_two);


	Mat gy(src.rows, src.cols, CV_32FC1);
	this->conv2_h_sobel<uchar>(       src, gy, this->sobel_two);
	this->conv2_v_sobel<float>(gy.clone(), gy, this->sobel_one);

#ifdef DEBUG_IMSHOW_RESULT
	Mat gy_show, gx_show;
	gy.convertTo(gy_show, CV_8UC1);
	gx.convertTo(gx_show, CV_8UC1);
	imshow("conv2_sobel() G(y) in 8-bit (from float)", gy_show);
	imshow("conv2_sobel() G(x) in 8-bit (from float)", gx_show);
	waitKey(10);
#endif 

	this->calculate_Magnitude(gx, gy);
	this->calculate_Gradients(gx, gy);

	gx.release();
	gy.release();

	this->nonMaxSuppresion(this->magnitude, this->gradient);

	return this->hysteresis_threshold(suppressed, high_thres, low_thres);

}


inline void Edge::calculate_Magnitude(const Mat& src1, const Mat& src2) {
	if(this->magnitude.empty()) this->magnitude = Mat(src1.rows, src1.cols, CV_32FC1);

#ifndef USE_SIMPLE_LOOP
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), this->magnitude.begin<float>(), 
		[](const float &s1, const float &s2) 
		{
			return std::sqrt(s1*s1 + s2*s2);
		}
	);
#else
	for (int i = 0; i < src1.rows; i++) {
		const float* gx = src1.ptr<float>(i);
		const float* gy = src2.ptr<float>(i);
		     float* dst = this->magnitude.ptr<float>(i);

		for (int j = 0; j < src1.cols; j++) {
			dst[j] = std::sqrt(gy[j] * gy[j] + gx[j] * gx[j]);
		}
	}
#endif


#ifdef DEBUG_IMSHOW_RESULT
	Mat magnitude_show;
	this->magnitude.convertTo(magnitude_show, CV_8UC1);
	imshow("calculate_magnitude() result in 8-bit (from float)", magnitude_show);
	waitKey(10);
#endif 

}


inline void Edge::calculate_Gradients(const Mat& src1, const Mat& src2) {
	
	if(this->gradient.empty()) this->gradient = Mat(src1.rows, src1.cols, CV_32FC1);

#ifndef USE_SIMPLE_LOOP
	// src2 = G(y) & src1 = G(x)
	std::transform(src1.begin<float>(), src1.end<float>(), src2.begin<float>(), this->gradient.begin<float>(),
		[](const float& gx, const float& gy)
		{
			if (gx[j] == 0 && gy[j] != 0)
				return 90;
			else if (gx[j] == 0 && gy[j] == 0)
				return 0;
			else {
				return (std::atan(gy[j] / gx[j]) * TO_THETA);
			}
		);

#else

	// Looping is faster than std::transform
	for (int i = 0; i < src1.rows; i++) {
		const float* gx = src1.ptr<float>(i);
		const float* gy = src2.ptr<float>(i);    
		float*      dst = this->gradient.ptr<float>(i);

		// Two if statement to improve speed, atan() is expensive
		for (int j = 0; j < src1.cols; j++) {
			if      (gx[j] == 0 && gy[j] != 0)
				dst[j] = 90;
			else if (gx[j] == 0 && gy[j] == 0)
				dst[j] = 0;
			else {
#ifdef SHOW_GRADIENT_RESULT
				cout << std::atan(gy[j] / gx[j]) << " : y=" << gy[j] << ", x=" << gx[j] << endl;
#endif
				dst[j] = std::atan(gy[j] / gx[j]) * TO_THETA;
			}
		}
	}
#endif // USE_SIMPLE_LOOPDEBUG


#ifdef DEBUG_IMSHOW_RESULT
	Mat gradient_show;
	this->gradient.convertTo(gradient_show, CV_8UC1);
	imshow("calculate_gradient() result in 8-bit (from float)", gradient_show);
	waitKey(10);
#endif 

	return;
}


void Edge::nonMaxSuppresion(Mat &magnitude, const Mat &gradient) {
	// Both magnitude & gradient are in float type
	if(this->suppressed.empty()) this->suppressed = Mat (magnitude.rows, magnitude.cols, CV_32FC1);
	int rows = magnitude.rows,
		cols = magnitude.cols;

	for (int i = 1; i < rows-1; i++) {
			  float* dst_ptr = this->suppressed.ptr<float>(i);
		      float* mag_ptr = magnitude.ptr<float>(i);
		const float* gra_ptr = gradient.ptr<float>(i);
		
		for (int j = 1; j < cols-1; j++) {
			int         theta = gra_ptr[j];
			float cur_mag_val = mag_ptr[j];
			theta = (theta < 0) ? 180 + theta : theta;

#ifdef NonMaxSuppress_SHOW_THETA_and_DIRECTIONS
			cout << "Theta = " << theta;
#endif 

			if (cur_mag_val != 0) {
				if (theta >= 67 && theta < 112) {
					// vertical direction
					if (cur_mag_val > mag_ptr[j - cols] && cur_mag_val > mag_ptr[j + cols])
						dst_ptr[j] = cur_mag_val;
#ifdef NonMaxSuppress_SHOW_THETA_and_DIRECTIONS
					cout << " -> [Vertical]" << endl;
#endif
				}
				else if ((theta <= 22 && theta >= 0) || (theta <= 180 && theta > 157)) {
					// horizontal direction
					if (cur_mag_val > mag_ptr[j - 1] && cur_mag_val > mag_ptr[j + 1])
						dst_ptr[j] = cur_mag_val;
#ifdef NonMaxSuppress_SHOW_THETA_and_DIRECTIONS
					cout << " -> [Horizontal]" << endl;
#endif
				}
				else if ((theta <= 157 && theta >= 112)) {
					// bottom-left to top-right direction
					if (cur_mag_val > mag_ptr[j + cols - 1] && cur_mag_val > mag_ptr[j - cols + 1])
						dst_ptr[j] = cur_mag_val;
#ifdef NonMaxSuppress_SHOW_THETA_and_DIRECTIONS
					cout << " -> [bl-tr]" << endl;
#endif
				}
				else if ((theta < 67 && theta > 22)) {
					// bottom-right to top-left direction
					if (cur_mag_val > mag_ptr[j + cols + 1] && cur_mag_val > mag_ptr[j - cols - 1])
						dst_ptr[j] = cur_mag_val;
#ifdef NonMaxSuppress_SHOW_THETA_and_DIRECTIONS
					cout << " -> [br-tl]" << endl;
#endif
				}
				else {
					// assign NaN to zero
					mag_ptr[j] = 0;
#ifdef NonMaxSuppress_SHOW_THETA_and_DIRECTIONS
					cout << " -> [I shouldn't exist]" << endl;
#endif
				}
			} 
			else {
				dst_ptr[j] = 0;
#ifdef NonMaxSuppress_SHOW_THETA_and_DIRECTIONS
				cout << " -> [Mag == 0]" << endl;
#endif
			}
		}
	}


#ifdef DEBUG_IMSHOW_RESULT
	Mat suppressed_show;
	this->suppressed.convertTo(suppressed_show, CV_8UC1);
	imshow("Edge class's nonMaxSuppression() result in 8-bit (from float)", suppressed_show);
	waitKey(10);
#endif 

}


Mat Edge::hysteresis_threshold(Mat& src, float high_thres, float low_thres) {

	if (src.empty() || src.channels() == 3) { cout << "hysteresis_threshold() error!\n"; return Mat(0,0,CV_8UC1); }

#ifndef USE_SIMPLE_LOOP
	std::transform(src.begin<float>(), src.end<float>(), src.begin<float>(), 
		[&high_thres, &low_thres] (const float &src_val) {
			if (src_val >= high_thres) 
				return 255; // Assign as strong edge pixel
			else if (src_val < high_thres && src_val >= low_thres)
				return 125; // Assign as potential strong edge pixel
			else 
				return 0;   // Suppressed to zero
		});

#else

	for (int i = 0; i < src.rows; i++) {
		float* src_ptr = src.ptr<float>(i);
		for (int j = 0; j < src.cols; j++) {
			if (src_ptr[j] >= high_thres)
				src_ptr[j] = 255;
			else if (src_ptr[j] < high_thres && src_ptr[j] >= low_thres)
				src_ptr[j] = 125;
			else
				src_ptr[j] = 0;
		}
	}
#endif // !USE_SIMPLE_LOOP



	// After thresholding, if pixel is assigned as 125,
	// check that pixel's 8-neighbor to see if any strong
	// pixel with intensity value of 255 exists. If so,
	// that pixel can be assigned as a strong pixel with
	// 255 intensity value. Otherwise, suppress it to 0
	Mat dst(src.rows, src.cols, CV_8UC1);
	int cols = src.cols;
	for (int i = 1; i < src.rows-1; i++) {
		uchar* neighbor_result    = dst.ptr<uchar>(i);
		float* double_thresholded = src.ptr<float>(i);
			
		for (int j = 1; j < cols-1; j++) {
			if (double_thresholded[j] == 0)        // No edge found
			{
				neighbor_result[j] = 0;
			}
			else if (double_thresholded[j] == 125) // potential strong edge pixel
			{ 
				if      (double_thresholded[j - 1]        == 255) neighbor_result[j] = 255;
				else if (double_thresholded[j + 1]        == 255) neighbor_result[j] = 255;
				else if (double_thresholded[j - cols]     == 255) neighbor_result[j] = 255;
				else if (double_thresholded[j + cols]     == 255) neighbor_result[j] = 255;
				else if (double_thresholded[j - cols - 1] == 255) neighbor_result[j] = 255;
				else if (double_thresholded[j - cols + 1] == 255) neighbor_result[j] = 255;
				else if (double_thresholded[j + cols + 1] == 255) neighbor_result[j] = 255;
				else if (double_thresholded[j + cols - 1] == 255) neighbor_result[j] = 255;
			}
			else  // Is a strong edge pixel
			{
				neighbor_result[j] = 255;
			}
		}
	}



#ifdef DEBUG_IMSHOW_RESULT
	imshow("Edge class's hysteresis_threshold() result in 8-bit (from float)", dst);
	waitKey(10);
#endif 

	return dst;
}