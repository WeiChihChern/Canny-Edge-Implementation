#include "Edge.h"

#include <algorithm>
#include <math.h>


Edge::Edge()
{
}


Edge::~Edge()
{
}









void Edge::CannyEdge(Mat& src, Mat &dst, float high_thres, float low_thres) {

	Mat copy1, copy2;
	this->conv2<uchar, short>(src, copy1, sobel_horizontal);
	this->conv2<uchar, short>(src, copy2, sobel_vertical);

	this->calculate_Magnitude<short,short>(copy1, copy2, true);
	this->calculate_Gradients<short, short>(copy1, copy2);
	
	copy1.release();
	copy2.release();

	this->nonMaxSuppresion(magnitude, gradient);

	magnitude.release();
	gradient.release();

	dst = this->hysteresis_threshold(suppressed, high_thres, low_thres);

	suppressed.release();

	return;
}












void Edge::cannyEdge2(Mat& src, Mat&dst, float high_thres, float low_thres) {

	Mat gx(src.rows, src.cols, CV_16SC1); // Short type
	this->conv2_h_sobel<uchar, short>(       src, gx, this->sobel_one);
	this->conv2_v_sobel<short, short>(gx.clone(), gx, this->sobel_two);

	
	Mat gy(src.rows, src.cols, CV_16SC1); // Short type
	this->conv2_h_sobel<uchar, short>(       src, gy, this->sobel_two);
	this->conv2_v_sobel<short, short>(gy.clone(), gy, this->sobel_one);

#ifdef DEBUG_IMSHOW_RESULT
	Mat gy_show, gx_show;
	gy.convertTo(gy_show, CV_8UC1);
	gx.convertTo(gx_show, CV_8UC1);
	imshow("conv2_sobel() G(y) in 8-bit (from float)", gy_show);
	imshow("conv2_sobel() G(x) in 8-bit (from float)", gx_show);
	waitKey(10);
#endif 

	

	// Save magnitude result in unsigned char (uchar) 
	this->calculate_Magnitude<short, short>(gx, gy, true);
	// Save gradient result in signed char (schar)
	this->calculate_Gradients<short, short>(gx, gy);

	gx.release();
	gy.release();

	this->nonMaxSuppresion(this->magnitude, this->gradient);

	dst = this->hysteresis_threshold(suppressed, high_thres, low_thres);

	magnitude.release();
	gradient.release();
	suppressed.release();

	return;

}












void Edge::nonMaxSuppresion(Mat &magnitude, const Mat &gradient) {
	// Both magnitude & gradient are in float type
	if(this->suppressed.empty()) this->suppressed = Mat (magnitude.rows, magnitude.cols, CV_8UC1, Scalar(0));
	int rows = magnitude.rows,
	    cols = magnitude.cols;

	OMP_FOR(rows * cols) // Automatically ignored if no openmp support
	for (int i = 1; i < rows-1; i++) {
		      uchar* dst_ptr = this->suppressed.ptr<uchar>(i);
		      uchar* mag_ptr = magnitude.ptr<uchar>(i);
		const schar* gra_ptr = gradient.ptr<schar>(i);
		
		for (int j = 1; j < cols-1; j++) {
			short       theta = (gra_ptr[j] < 0) ? 180+ gra_ptr[j] : gra_ptr[j];
			uchar cur_mag_val = mag_ptr[j];

#ifdef DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS
			bool suppressed_to_zero = false;
			cout << "( Theta = " << theta << " , Magnitude = " << (int)cur_mag_val << " )";
#endif 

			if (cur_mag_val != 0) // Edge pixel
			{ 
				if (theta >= 67 && theta <= 112) 
				{
					// vertical direction
					if ( cur_mag_val > mag_ptr[j - cols] && cur_mag_val >= mag_ptr[j + cols] ) {
						dst_ptr[j] = cur_mag_val;
					} 
#ifdef DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS
					else {
						suppressed_to_zero = true;
					}
					cout << " -> [Check vertical : (j-cols = " << (int)mag_ptr[j - cols] << " , ";
					cout << "j+cols = " << (int)mag_ptr[j + cols] << ") ";
					cout << "Action -> ";
					if (suppressed_to_zero)
						cout << "[Suppressed to Zero]\n";
					else
						cout << "[Not suppressed]\n";		
#endif
				}
				else if ((theta < 23 && theta >= 0) || (theta <= 180 && theta > 157)) 
				{
					// horizontal direction
					if (cur_mag_val > mag_ptr[j - 1] && cur_mag_val >= mag_ptr[j + 1]) {
						dst_ptr[j] = cur_mag_val;
					}
#ifdef DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS
					else {
						suppressed_to_zero = true;
					}
					cout << " -> [Check horizontal : (j-1 = " << (int)mag_ptr[j - 1] << " , ";
					cout << "j+1 = " << (int)mag_ptr[j + 1] << ") ";
					cout << "Action -> ";
					if (suppressed_to_zero)
						cout << "[Suppressed to Zero]\n";
					else
						cout << "[Not suppressed]\n";
#endif
				}
				else  // bottom-left to top-right  or  bottom-right to top-left direction
				{ 
					int d = (theta > 90) ? 1 : -1;
					if (cur_mag_val >= mag_ptr[j + cols - d] && cur_mag_val > mag_ptr[j - cols + d]) {
						dst_ptr[j] = cur_mag_val;
					}
#ifdef DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS
					else {
						suppressed_to_zero = true;
					}
					if (d == 1) {
						cout << " -> [Check bl-tr : (";
						cout << "j+cols-1 = " << (int)mag_ptr[j + cols - 1] << " , ";
						cout << "j-cols+1 = " << (int)mag_ptr[j - cols + 1] << ") ";
					}
					else {
						cout << " -> [Check br-tl : (";
						cout << "j+cols+1 = " << (int)mag_ptr[j + cols + 1] << " , ";
						cout << "j-cols-1 = " << (int)mag_ptr[j - cols - 1] << ") ";
					}
					cout << "Action -> ";
					if (suppressed_to_zero)
						cout << "[Suppressed to Zero]\n";
					else
						cout << "[Not suppressed]\n";
#endif
				}
			} 
			else // Non edge pixel
			{ 
				dst_ptr[j] = 0;
#ifdef DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS
				cout << " -> [Mag == 0]" << endl;
#endif
			}
		}
	}


#ifdef DEBUG_IMSHOW_RESULT
	imshow("Edge class's nonMaxSuppression() result in 8-bit (from float)", this->suppressed);
	waitKey(0);
#endif 

	return;
}














Mat Edge::hysteresis_threshold(Mat& src, float high_thres, float low_thres) {

	if (src.empty() || src.channels() == 3) { cout << "hysteresis_threshold() error!\n"; return Mat(0,0,CV_8UC1); }

	int rows = src.rows,
	    cols = src.cols;

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
	

	OMP_FOR(rows * cols) // Automatically ignored if no openmp support
	for (int i = 0; i < src.rows; i++) 
	{
		uchar* src_ptr = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++) 
		{
			uchar* val = (src_ptr + j);
			if (*val >= high_thres)
				* val = 255;
			else if (*val < high_thres && *val >= low_thres)
				*val = 125;
			else
				*val = 0;
		}
	}

#endif // !USE_SIMPLE_LOOP



	// After thresholding, if pixel is assigned as 125,
	// check that pixel's 8-neighbor to see if any strong
	// pixel with intensity value of 255 exists. If so,
	// that pixel can be assigned as a strong pixel with
	// 255 intensity value. Otherwise, suppress it to 0
	Mat dst(src.rows, src.cols, CV_8UC1, Scalar(0));
	
	OMP_FOR(rows * cols) // Automatically ignored if no openmp support
	for (int i = 1; i < rows-1; i++) 
	{
		uchar* neighbor_result    = dst.ptr<uchar>(i);
		uchar* double_thresholded = src.ptr<uchar>(i);
			
		for (int j = 1; j < cols-1; j++) 
		{
			uchar val = double_thresholded[j];
			if (val == 0)        // No edge found
			{
				neighbor_result[j] = 0;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
				cout << "Not a edge pixel, no checking needed.\n";
#endif 
			}
			else if (val == 125) // potential strong edge pixel
			{
				if (*(double_thresholded + j - 1) == 255) {
					neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
					cout << "Neighbor 255 is at: j - 1\n";
#endif 
				}
				else if (*(double_thresholded + j + 1) == 255) {
					neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
					cout << "Neighbor 255 is at: j + 1\n";
#endif 
				}
				else if (*(double_thresholded+j - cols) == 255) {
					neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
					cout << "Neighbor 255 is at: j - cols\n";
#endif 
				}
				else if (*(double_thresholded + j + cols) == 255) {
					neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
					cout << "Neighbor 255 is at: j + cols\n";
#endif 
				}
				else if (*(double_thresholded + j - cols - 1) == 255) {
					neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
					cout << "Neighbor 255 is at: j - cols - 1\n";
#endif 
				}
				else if (*(double_thresholded + j - cols + 1) == 255) {
				neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
				cout << "Neighbor 255 is at: j + cols + 1\n";
#endif 
				}
				else if (*(double_thresholded + j + cols + 1) == 255) {
					neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
					cout << "Neighbor 255 is at: j + cols + 1\n";
#endif 
				}
				else if (*(double_thresholded + j + cols - 1) == 255) {
					neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
					cout << "Neighbor 255 is at: j + cols - 1\n";
#endif 
				}
				else { // No strong pixel (=255) in 8 neighbors
					neighbor_result[j] = 0;
				}

			}
			else  // Is a strong edge pixel
			{
				neighbor_result[j] = 255;
#ifdef DEBUG_SHOW_HYSTERESIS_NEIGHBOR_RESULT
				cout << "Strong edge pixel, no checking needed.\n";
#endif 
			}
		}
	}



#ifdef DEBUG_IMSHOW_RESULT
	imshow("Edge class's hysteresis_threshold() result in 8-bit (from float)", dst);
	waitKey(10);
#endif 



	return dst;
}




inline double Edge::FastArcTan(double x)
{
	return 0.785398163397448309616*x - x*(fabs(x) - 1)*(0.2447 + 0.0663*fabs(x));
}
