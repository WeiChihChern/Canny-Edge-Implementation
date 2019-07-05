#include <iostream>
#include <algorithm>

#include "cublas_v2.h"
#include "Canny_Cuda.cuh"
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"


using namespace std;

#define num_threads 16



// Should only allocate 1D blocks with threads
template <typename inputType>
__global__ void conv2_h_sobel_cuda_function(
	const inputType* data, float* dst, int rows, int cols, float* kernel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int cur_row = idx / cols;
	int cur_col = idx % cols;
	
	if (cur_col > 0 && cur_col < cols - 1)
		dst[idx] = data[idx - 1]  * *kernel + data[idx]     * *(kernel+1) + data[idx + 1] * *(kernel+2);

	return;
}


template <typename inputType>
__global__ void conv2_v_sobel_cuda_function(
	const inputType *data, float *dst, int rows, int cols, float* kernel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int cur_row = idx / cols;
	int cur_col = idx % cols;

	if (cur_row > 0 && cur_row < rows - 1)
		dst[idx] = data[idx - cols]  * *kernel + data[idx] * *(kernel + 1) + data[idx + cols] * *(kernel + 2);

	return;
}



// This magnitude function will store the result in gx
template <typename inputType>
__global__ void calculate_magnitude_cuda_function(
	const inputType *gy, const inputType *gx, inputType *dst, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int cur_row = idx / cols;
	int cur_col = idx % cols;

	if (cur_col > 0 && cur_col < cols - 1 && 
		cur_row > 0 && cur_row < rows - 1)
		dst[idx] = std::sqrt(gy[idx] + gx[idx]);

	return;
}



template <typename inputType>
__global__ void gradient_cuda_function(const inputType* gx, const inputType* gy, int rows, int cols, inputType* dst)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int cur_row = idx / cols;
	int cur_col = idx % cols;

	if (cur_col > 0 && cur_col < cols - 1 && 
		cur_row > 0 && cur_row < rows - 1)
	{
			float w = abs(gy[idx] / (gx[idx]+0.0001));

			if (w < 0.4)
				dst[idx] = 0;
			else if (w > 2.3)
				dst[idx] = 90;
			else 
				dst[idx] = 45;
	}

}





template <typename inputType>
__global__ void nonMax_cuda_function(
	const inputType* magnitude, const inputType* gradient, 
	const inputType *gx, const inputType *gy, inputType* dst, 
	int rows, int cols, float high_thres, float low_thres)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int cur_row = idx / cols;
	int cur_col = idx % cols;

	inputType cur_magnitude, theta;


	if (cur_col > 0 && cur_col < cols - 1 && 
		cur_row > 0 && cur_row < rows - 1)
	{
		cur_magnitude = magnitude[idx];
		theta         = gradient[idx];

		if ( cur_magnitude > low_thres && cur_magnitude != 0 ) // Edge pixel
		{ 
				if (theta == 90) 
				{
						// vertical direction
						if ( cur_magnitude > magnitude[idx - cols] && cur_magnitude >= magnitude[idx + cols] ) 
							dst[idx] = (cur_magnitude >= high_thres) ? 255 : cur_magnitude;
				}
				else if (theta == 0) 
				{
						// horizontal direction
						if (cur_magnitude > magnitude[idx - 1] && cur_magnitude >= magnitude[idx + 1]) 
							dst[idx] = (cur_magnitude >= high_thres) ? 255 : cur_magnitude;
				}
				else  if (theta == 45)// bottom-left to top-right  or  bottom-right to top-left direction
				{ 
						int d = (gy[idx] * gx[idx] < 0) ? 1 : -1;
						if (cur_magnitude >= magnitude[idx + cols - d] && cur_magnitude > magnitude[idx - cols + d]) 
							dst[idx] = (cur_magnitude >= high_thres) ? 255 : cur_magnitude;
				}
				else 
							dst[idx] = 0;
		}
		else // Non edge pixel
				dst[idx] = 0; 
	}
}



__global__ void normalizedTo(
	float *src, size_t n, int max_idx, int min_idx, float range, float range2, float high_val,
	float low_val)
{
	float min_val = src[min_idx];


	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= 0 && idx < n)
		src[idx] = ((src[idx] - min_val) / range) * range2 + low_val;
	
}



template <typename inputType>
__global__ void hysteresis_cuda_function(
	inputType* src, int rows, int cols, float high_thres, float low_thres)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int cur_row = idx / cols;
	int cur_col = idx % cols;


	if (cur_col > 0 && cur_col < cols - 1 && 
		cur_row > 0 && cur_row < rows - 1)
	{
			if(src[idx] < high_thres && src[idx] > low_thres)
			{
						if (*(src + idx - 1)        == 255 || *(src + idx + 1)        == 255 || *(src+idx - cols) == 255 || 
							*(src + idx + cols)     == 255 || *(src + idx - cols - 1) == 255 || 
							*(src + idx - cols + 1) == 255 || *(src + idx + cols + 1) == 255 || *(src + idx + cols - 1) == 255) 
						{
							src[idx] = 255;
						}
						else // No strong pixel (=255) in 8 neighbors
						{ 
							src[idx] = 0;
						}
			}	
	}
}



template <typename inputType>
__global__ void makeZero(inputType* src, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int cur_row = idx / cols;
	int cur_col = idx % cols;

	if(cur_col >= 0 && cur_col < cols &&
	   cur_row >= 0 && cur_row < rows)
	   src[idx] = 0;

}













extern "C"
void conv2_sobel_cuda(unsigned char* src, float* dst, int r, int c, float high_thres, float low_thres)
{

	// Initializing
	int	size = r*c;

	kernel kernel;

	unsigned char* gpu_src;
	float 	
		*gpu_dst, *gpu_dst_gy, *gpu_dst_gx,
		*gpu_kernel1, *gpu_kernel2,
		*gpu_magnitude, *gpu_gradient,
		*gpu_nonMax;
	cudaError_t  error, cudaStatus;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}





	//
	//
	// Allocating memory on gpu, and copy cpu memory to gpu
	error = cudaMalloc((void**)&gpu_src, size * sizeof(unsigned char));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	error = cudaMalloc((void**)&gpu_dst, size * sizeof(float));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	error = cudaMalloc((void**)&gpu_dst_gy, size * sizeof(float));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	error = cudaMalloc((void**)&gpu_dst_gx, size * sizeof(float));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	error = cudaMalloc((void**)&gpu_kernel1, 3 * sizeof(float));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	error = cudaMalloc((void**)&gpu_kernel2, 3 * sizeof(float));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	error = cudaMalloc((void**)&gpu_magnitude, size * sizeof(float));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	error = cudaMalloc((void**)&gpu_gradient, size * sizeof(float));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	error = cudaMalloc((void**)&gpu_nonMax, size * sizeof(float));
	if (error != cudaSuccess) {
		cout << "cudaMalloc failed\n";
		goto Error;
	}
	



	//
	//
	// Copy data from host to device
	error = cudaMemcpy(gpu_src, src, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cout << "cudaMemcopy failed\n";
		goto Error;
	}
	error = cudaMemcpy(gpu_kernel1, kernel.k1, 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cout << "cudaMemcopy failed\n";
		goto Error;
	}
	error = cudaMemcpy(gpu_kernel2, kernel.k2, 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cout << "cudaMemcopy failed\n";
		goto Error;
	}
	


	//
	//
	// cuda functions were written in taking 1D blocks
	// so don't create a 2D blocks
	dim3 threadsPerBlock(num_threads, 1);
	dim3 blocksPerGrid(size / num_threads + 1, 1);

	// Get gy
	conv2_h_sobel_cuda_function<<< blocksPerGrid, threadsPerBlock>>>(gpu_src,  gpu_dst,   r, c, gpu_kernel1);
	conv2_v_sobel_cuda_function<<< blocksPerGrid, threadsPerBlock>>>(gpu_dst, gpu_dst_gy, r, c, gpu_kernel2);


	// Get gx
	conv2_h_sobel_cuda_function<<<blocksPerGrid, threadsPerBlock>>>(gpu_src,  gpu_dst,   r, c, gpu_kernel2);
	conv2_h_sobel_cuda_function<<<blocksPerGrid, threadsPerBlock>>>(gpu_dst, gpu_dst_gx, r, c, gpu_kernel1);


	// Get magnitude result
	makeZero<<<blocksPerGrid, threadsPerBlock>>>(gpu_magnitude, r, c);
	calculate_magnitude_cuda_function<<<blocksPerGrid, threadsPerBlock>>>(gpu_dst_gy, gpu_dst_gx, gpu_magnitude, r, c);


	int max_idx, min_idx;
	cublasHandle_t handle;
	cublasIsamax(handle, size, gpu_magnitude, 1, &max_idx);
	cublasIsamin(handle, size, gpu_magnitude, 1, &min_idx);
	
	float range = src[max_idx] - src[min_idx];
	
	normalizedTo<<<blocksPerGrid, threadsPerBlock>>>(gpu_magnitude, size, max_idx, min_idx, range, 255, 255, 0);

	
	
	// Get gradient result
	// gradient_cuda_function<<<blocksPerGrid, threadsPerBlock>>>(gpu_dst_gx, gpu_dst_gy, r, c, gpu_gradient);



	// Get nonMax result
	// nonMax_cuda_function<<<blocksPerGrid, threadsPerBlock>>>(
	// 	gpu_magnitude, gpu_gradient, gpu_dst_gx, gpu_dst_gy, gpu_nonMax, r, c, high_thres, low_thres);

	

	// Get threshold result
	//hysteresis_cuda_function<<<blocksPerGrid, threadsPerBlock>>>(gpu_nonMax, r, c, high_thres, low_thres);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}



	
	// Copy the result back to cpu 
	error = cudaMemcpy(dst, gpu_magnitude, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		cout << "cudaMemcopy failed\n";
		goto Error;
	}



Error:
	cudaFree(gpu_dst);
	cudaFree(gpu_src);
	cudaFree(gpu_kernel1);
	cudaFree(gpu_kernel2);


	cudaFree(gpu_dst_gy);
	cudaFree(gpu_dst_gx);
	cudaFree(gpu_magnitude);
	cudaFree(gpu_gradient);
	cudaFree(gpu_nonMax);


	return;
}














