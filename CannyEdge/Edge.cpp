#include "Edge.h"

#include <algorithm>
#define _USE_MATH_DEFINES // for C++
#include <math.h>




Edge::Edge()
{
}


Edge::~Edge()
{
}









void Edge::CannyEdge(Mat& src, Mat &dst, float high_thres, float low_thres) {

	this->rows = src.rows;
	this->cols = src.cols;
	this->size = this->rows * this->cols;

	Mat copy1, copy2;
	this->conv2<uchar, short>(src, copy1, sobel_horizontal);
	this->conv2<uchar, short>(src, copy2, sobel_vertical);

	this->calculate_Magnitude<short,short>(copy1, copy2, true);
	this->calculate_Gradients<short, short>(copy1, copy2);
	
	copy1.release();
	copy2.release();

	this->nonMaxSuppresion(magnitude, gradient, high_thres, low_thres);

	magnitude.release();
	gradient.release();

	dst = this->hysteresis_threshold(suppressed);

	suppressed.release();

	return;
}












void Edge::cannyEdge2(Mat& src, Mat&dst, float high_thres, float low_thres) {

	this->rows = src.rows;
	this->cols = src.cols;
	this->size = this->rows * this->cols;

#ifdef _OPENMP
	omp_set_num_threads(threadControl(this->size));
#endif


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

	this->nonMaxSuppresion(this->magnitude, this->gradient, high_thres, low_thres);

	dst = this->hysteresis_threshold(suppressed);

	magnitude.release();
	gradient.release();
	suppressed.release();

	return;

}












void Edge::nonMaxSuppresion(const Mat &magnitude, const Mat &gradient, float high_thres, float low_thres) {
	// Both magnitude & gradient are in float type
	if(this->suppressed.empty()) this->suppressed = Mat (this->rows, this->cols, CV_8UC1, Scalar(0)); //remove scalar 0 to optimize



	#pragma omp parallel for 
	for (size_t i = 1; i < this->rows-1; ++i) {
		      uchar* dst_ptr = this->suppressed.ptr<uchar>(i);
		const uchar* mag_ptr = magnitude.ptr<uchar>(i);
		const schar* gra_ptr = gradient.ptr<schar>(i);

#ifdef __GNUC__
		#pragma omp simd
#endif
		for (size_t j = 1; j < this->cols-1; ++j) {
			short       theta = (*(gra_ptr+j) < 0) ? 180 + *(gra_ptr+j) : *(gra_ptr+j);
			uchar cur_mag_val = *(mag_ptr+j);

#ifdef DEBUG_SHOW_NonMaxSuppress_THETA_and_DIRECTIONS
			bool suppressed_to_zero = false;
			cout << "( Theta = " << theta << " , Magnitude = " << (int)cur_mag_val << " )";
#endif 

			if ( cur_mag_val > low_thres && cur_mag_val != 0 ) // Edge pixel
			{ 
				if (theta >= 67 && theta <= 112) 
				{
					// vertical direction
					if ( cur_mag_val > *(mag_ptr + j - cols) && cur_mag_val >= *(mag_ptr + j + cols) ) {
						dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : cur_mag_val;
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
					if (cur_mag_val > *(mag_ptr + j - 1) && cur_mag_val >= *(mag_ptr + j + 1)) {
						dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : cur_mag_val;
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
					if (cur_mag_val >= *(mag_ptr + j + cols - d) && cur_mag_val > *(mag_ptr + j - cols + d)) {
						dst_ptr[j] = (cur_mag_val >= high_thres) ? 255 : cur_mag_val;
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
	waitKey(10);
#endif 

	return;
}
















void Edge::new_nonMaxSuppression(const Mat& magnitude, const Mat &gradient)
{
    const int TG22 = 13573;
	if(this->suppressed.empty()) this->suppressed = Mat (this->rows, this->cols, CV_8UC1, Scalar(0));

}


















Mat Edge::hysteresis_threshold(const Mat& src) {

	if (src.empty() || src.channels() == 3) { cout << "hysteresis_threshold() error!\n"; return Mat(0,0,CV_8UC1); }



	// In nonMax(), not only find the max along the graident direction,
	// but everything >= high_thres is set to 255, and everything < low_thres
	// is set to zero. 
	// Check the value between high_thres & lower_thres to see if there's any 
	// strong pixel in the 8-neighbor, and set itself to 255 if it does.
	Mat dst(this->rows, this->cols, CV_8UC1); //Added scalar(0)
	
#pragma omp parallel for
	for (size_t i = 1; i < this->rows-1; ++i)
	{
		      uchar* dst_p  = dst.ptr<uchar>(i); 
		const uchar* nonM_p = src.ptr<uchar>(i); // non max result pointer
			
#ifdef __GNUC__
		#pragma omp simd // for -O2 optimization
#endif	
		for (size_t j = 1; j < this->cols-1; ++j) 
		{
			uchar val = nonM_p[j];
			if(val == 0)
				dst_p[j] = val;
			else if (val == 255)
				dst_p[j] = val;
			else
			{
				if (*(nonM_p + j - 1) == 255 || *(nonM_p + j + 1) == 255 || *(nonM_p+j - cols) == 255 || 
				    *(nonM_p + j + cols) == 255 || *(nonM_p + j - cols - 1) == 255 || 
					*(nonM_p + j - cols + 1) == 255 || *(nonM_p + j + cols + 1) == 255 || *(nonM_p + j + cols - 1) == 255) 
				{
					dst_p[j] = 255;
				}
				else { // No strong pixel (=255) in 8 neighbors
					dst_p[j] = 0;
				}
			}
			
		}
	}

	this->edge2zero<uchar>(dst);

#ifdef DEBUG_IMSHOW_RESULT
	imshow("Edge class's hysteresis_threshold() result in 8-bit (from float)", dst);
	waitKey(10);
#endif 



	return dst;
}






double Edge::FastArcTan(double x)
{
	return M_PI_4*x - x*(fabs(x) - 1)*(0.2447 + 0.0663*fabs(x));
}